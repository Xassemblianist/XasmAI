#pragma once

#include "matrix.hpp"
#include "attention.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace xasm {

class RMSNorm {
    size_t dim;
    Matrix gamma, dgamma;
    ParamState sg;
    Matrix x_norm_cache, rms_cache;
    double eps;
public:
    RMSNorm() : dim(0), eps(1e-6) {}
    explicit RMSNorm(size_t d, double e = 1e-6) : dim(d), eps(e) {
        gamma = Matrix::ones(1, d); sg.init(1, d); dgamma = Matrix::zeros(1, d);
    }
    Matrix forward(const Matrix& x) {
        size_t seq = x.rows;
        rms_cache = Matrix(seq, 1); x_norm_cache = Matrix(seq, dim);
        for (size_t i = 0; i < seq; ++i) {
            double sq = 0.0;
            for (size_t j = 0; j < dim; ++j) sq += x(i, j) * x(i, j);
            double rms = std::sqrt(sq / static_cast<double>(dim) + eps);
            rms_cache(i, 0) = rms;
            double inv = 1.0 / rms;
            for (size_t j = 0; j < dim; ++j) x_norm_cache(i, j) = x(i, j) * inv;
        }
        Matrix out(seq, dim);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = 0; j < dim; ++j) out(i, j) = gamma(0, j) * x_norm_cache(i, j);
        return out;
    }
    Matrix backward(const Matrix& d_out) {
        size_t seq = d_out.rows;
        double N = static_cast<double>(dim);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = 0; j < dim; ++j) dgamma(0, j) += d_out(i, j) * x_norm_cache(i, j);
        Matrix dx(seq, dim);
        for (size_t i = 0; i < seq; ++i) {
            double rms_v = rms_cache(i, 0), sum_dx = 0.0;
            for (size_t j = 0; j < dim; ++j) sum_dx += d_out(i, j) * gamma(0, j) * x_norm_cache(i, j);
            for (size_t j = 0; j < dim; ++j)
                dx(i, j) = (d_out(i, j) * gamma(0, j) - x_norm_cache(i, j) * sum_dx / N) / rms_v;
        }
        return dx;
    }
    void update_weights(Optimizer& opt) { opt.update(gamma, dgamma, sg); }
    void zero_grad() { dgamma = Matrix::zeros(1, dim); }
    void save(std::ostream& os) const { gamma.serialize(os); sg.serialize(os); }
    void load(std::istream& is) { gamma = Matrix::deserialize(is); sg.deserialize(is); dgamma = Matrix::zeros(1, dim); }
    size_t param_count() const { return dim; }
};

class SwiGLU {
    size_t d_model, d_ff;
    Matrix W_gate, W_up, W_down;
    Matrix dW_gate, dW_up, dW_down;
    ParamState sWg, sWu, sWd;
    Matrix input_cache, gate_pre_cache, up_cache, gate_cache;
    static double silu(double x) { return x / (1.0 + std::exp(-x)); }
    static double silu_d(double x) { double s = 1.0 / (1.0 + std::exp(-x)); return s * (1.0 + x * (1.0 - s)); }
public:
    SwiGLU() : d_model(0), d_ff(0) {}
    SwiGLU(size_t dm, size_t df) : d_model(dm), d_ff(df) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dg(0.0, std::sqrt(2.0 / static_cast<double>(dm)));
        std::normal_distribution<double> dd(0.0, std::sqrt(2.0 / static_cast<double>(df)));
        W_gate = Matrix(dm, df); W_up = Matrix(dm, df); W_down = Matrix(df, dm);
        for (auto& v : W_gate.data) v = dg(gen);
        for (auto& v : W_up.data) v = dg(gen);
        for (auto& v : W_down.data) v = dd(gen);
        sWg.init(dm, df); sWu.init(dm, df); sWd.init(df, dm);
        zero_grad();
    }
    Matrix forward(const Matrix& x) {
        input_cache = x; gate_pre_cache = x * W_gate; up_cache = x * W_up;
        size_t seq = x.rows;
        gate_cache = Matrix(seq, d_ff);
        Matrix hidden(seq, d_ff);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = 0; j < d_ff; ++j) {
                gate_cache(i, j) = silu(gate_pre_cache(i, j));
                hidden(i, j) = gate_cache(i, j) * up_cache(i, j);
            }
        return hidden * W_down;
    }
    Matrix backward(const Matrix& d_out) {
        size_t seq = d_out.rows;
        Matrix hidden(seq, d_ff);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = 0; j < d_ff; ++j) hidden(i, j) = gate_cache(i, j) * up_cache(i, j);
        dW_down += hidden.T() * d_out;
        Matrix d_h = d_out * W_down.T();
        Matrix d_gp(seq, d_ff), d_up(seq, d_ff);
        for (size_t i = 0; i < seq; ++i)
            for (size_t j = 0; j < d_ff; ++j) {
                d_up(i, j) = d_h(i, j) * gate_cache(i, j);
                d_gp(i, j) = d_h(i, j) * up_cache(i, j) * silu_d(gate_pre_cache(i, j));
            }
        dW_gate += input_cache.T() * d_gp; dW_up += input_cache.T() * d_up;
        return d_gp * W_gate.T() + d_up * W_up.T();
    }
    void update_weights(Optimizer& opt) {
        opt.update(W_gate, dW_gate, sWg); opt.update(W_up, dW_up, sWu); opt.update(W_down, dW_down, sWd);
    }
    void zero_grad() {
        dW_gate = Matrix::zeros(d_model, d_ff);
        dW_up = Matrix::zeros(d_model, d_ff);
        dW_down = Matrix::zeros(d_ff, d_model);
    }
    void save(std::ostream& os) const {
        W_gate.serialize(os); W_up.serialize(os); W_down.serialize(os);
        sWg.serialize(os); sWu.serialize(os); sWd.serialize(os);
    }
    void load(std::istream& is) {
        W_gate = Matrix::deserialize(is); W_up = Matrix::deserialize(is); W_down = Matrix::deserialize(is);
        sWg.deserialize(is); sWu.deserialize(is); sWd.deserialize(is); zero_grad();
    }
    size_t param_count() const { return 2 * d_model * d_ff + d_ff * d_model; }
};

class MoELayer {
    std::vector<SwiGLU> experts;
    Matrix W_router, dW_router;
    ParamState sR;
    size_t d_model, num_experts, active_k;
    double balance_coeff;
    Matrix input_cache, gate_probs_cache;
    struct ExpertBatch { std::vector<size_t> tids; std::vector<double> weights; };
    std::vector<ExpertBatch> batches;
    std::vector<Matrix> expert_out_cache;

public:
    MoELayer() : d_model(0), num_experts(0), active_k(0), balance_coeff(0) {}

    MoELayer(size_t dm, size_t d_ff, size_t ne, size_t ak, double bc = 0.01)
        : d_model(dm), num_experts(ne), active_k(ak), balance_coeff(bc) {
        for (size_t i = 0; i < ne; ++i) experts.emplace_back(dm, d_ff);
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 0.01);
        W_router = Matrix(dm, ne);
        for (auto& v : W_router.data) v = dist(gen);
        dW_router = Matrix::zeros(dm, ne); sR.init(dm, ne);
    }

    Matrix forward(const Matrix& x) {
        input_cache = x;
        size_t seq = x.rows;
        Matrix gl = x * W_router;

        gate_probs_cache = Matrix(seq, num_experts);
        for (size_t i = 0; i < seq; ++i) {
            double mx = gl(i, 0);
            for (size_t e = 1; e < num_experts; ++e) mx = std::max(mx, gl(i, e));
            double sm = 0.0;
            for (size_t e = 0; e < num_experts; ++e) {
                gate_probs_cache(i, e) = std::exp(gl(i, e) - mx);
                sm += gate_probs_cache(i, e);
            }
            for (size_t e = 0; e < num_experts; ++e) gate_probs_cache(i, e) /= sm;
        }

        batches.assign(num_experts, ExpertBatch{});
        for (size_t i = 0; i < seq; ++i) {
            std::vector<std::pair<double, size_t>> scores(num_experts);
            for (size_t e = 0; e < num_experts; ++e) scores[e] = {gate_probs_cache(i, e), e};
            std::partial_sort(scores.begin(), scores.begin() + static_cast<long>(active_k),
                               scores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t k = 0; k < active_k; ++k) {
                batches[scores[k].second].tids.push_back(i);
                batches[scores[k].second].weights.push_back(scores[k].first);
            }
        }

        expert_out_cache.resize(num_experts);
        Matrix output(seq, d_model, 0.0);

        for (size_t e = 0; e < num_experts; ++e) {
            if (batches[e].tids.empty()) { expert_out_cache[e] = Matrix(); continue; }
            size_t bs = batches[e].tids.size();
            Matrix batch(bs, d_model);
            for (size_t b = 0; b < bs; ++b)
                for (size_t j = 0; j < d_model; ++j) batch(b, j) = x(batches[e].tids[b], j);

            Matrix eo = experts[e].forward(batch);
            expert_out_cache[e] = eo;

            for (size_t b = 0; b < bs; ++b) {
                double w = batches[e].weights[b];
                size_t tid = batches[e].tids[b];
                for (size_t j = 0; j < d_model; ++j) output(tid, j) += w * eo(b, j);
            }
        }
        return output;
    }

    double get_balance_loss() const {
        size_t seq = gate_probs_cache.rows;
        if (seq == 0) return 0.0;
        double total = static_cast<double>(seq * active_k), loss = 0.0;
        for (size_t e = 0; e < num_experts; ++e) {
            double f = static_cast<double>(batches[e].tids.size()) / total;
            double P = 0.0;
            for (size_t i = 0; i < seq; ++i) P += gate_probs_cache(i, e);
            P /= static_cast<double>(seq);
            loss += f * P;
        }
        return loss * static_cast<double>(num_experts) * balance_coeff;
    }

    Matrix backward(const Matrix& d_output) {
        size_t seq = d_output.rows;
        Matrix d_input(seq, d_model, 0.0);
        Matrix d_gp(seq, num_experts, 0.0);

        for (size_t e = 0; e < num_experts; ++e) {
            if (batches[e].tids.empty()) continue;
            size_t bs = batches[e].tids.size();

            Matrix d_eo(bs, d_model);
            for (size_t b = 0; b < bs; ++b) {
                double w = batches[e].weights[b];
                size_t tid = batches[e].tids[b];
                for (size_t j = 0; j < d_model; ++j) d_eo(b, j) = w * d_output(tid, j);
            }

            Matrix d_ei = experts[e].backward(d_eo);

            for (size_t b = 0; b < bs; ++b) {
                size_t tid = batches[e].tids[b];
                for (size_t j = 0; j < d_model; ++j) d_input(tid, j) += d_ei(b, j);
                double dw = 0.0;
                for (size_t j = 0; j < d_model; ++j)
                    dw += d_output(tid, j) * expert_out_cache[e](b, j);
                d_gp(tid, e) = dw;
            }
        }

        double total = static_cast<double>(seq * active_k);
        for (size_t e = 0; e < num_experts; ++e) {
            double f = static_cast<double>(batches[e].tids.size()) / total;
            for (size_t i = 0; i < seq; ++i)
                d_gp(i, e) += balance_coeff * f * static_cast<double>(num_experts) /
                               static_cast<double>(seq);
        }

        Matrix d_gl(seq, num_experts);
        for (size_t i = 0; i < seq; ++i) {
            double dot = 0.0;
            for (size_t e = 0; e < num_experts; ++e) dot += gate_probs_cache(i, e) * d_gp(i, e);
            for (size_t e = 0; e < num_experts; ++e)
                d_gl(i, e) = gate_probs_cache(i, e) * (d_gp(i, e) - dot);
        }
        dW_router += input_cache.T() * d_gl;
        d_input += d_gl * W_router.T();
        return d_input;
    }

    void update_weights(Optimizer& opt) {
        opt.update(W_router, dW_router, sR);
        for (auto& e : experts) e.update_weights(opt);
    }
    void zero_grad() {
        dW_router = Matrix::zeros(d_model, num_experts);
        for (auto& e : experts) e.zero_grad();
    }
    void save(std::ostream& os) const {
        W_router.serialize(os); sR.serialize(os);
        for (const auto& e : experts) e.save(os);
    }
    void load(std::istream& is) {
        W_router = Matrix::deserialize(is); sR.deserialize(is);
        for (auto& e : experts) e.load(is);
        dW_router = Matrix::zeros(d_model, num_experts);
    }
    size_t param_count() const {
        size_t t = d_model * num_experts;
        for (const auto& e : experts) t += e.param_count();
        return t;
    }
};

class TransformerBlock {
    RMSNorm ln1, ln2;
    MultiHeadAttention mha;
    SwiGLU single_ffn;
    MoELayer moe;
    bool use_moe, grad_ckpt;
    Matrix block_input_cache;

public:
    TransformerBlock() : use_moe(false), grad_ckpt(false) {}

    TransformerBlock(size_t dm, size_t nh, size_t nkv, size_t d_ff,
                      size_t max_seq, size_t ws, size_t n_experts,
                      size_t active_exp, double bal_c, bool gc)
        : ln1(dm), ln2(dm), mha(dm, nh, nkv, max_seq, ws),
          use_moe(n_experts > 1), grad_ckpt(gc) {
        if (use_moe) moe = MoELayer(dm, d_ff, n_experts, active_exp, bal_c);
        else single_ffn = SwiGLU(dm, d_ff);
    }

    Matrix forward(const Matrix& x, size_t start_pos = 0, bool use_cache = false) {
        if (grad_ckpt && !use_cache) block_input_cache = x;
        Matrix z = x + mha.forward(ln1.forward(x), start_pos, use_cache);
        Matrix ffn_out = use_moe ? moe.forward(ln2.forward(z)) : single_ffn.forward(ln2.forward(z));
        return z + ffn_out;
    }

    Matrix backward(const Matrix& d_out) {
        if (grad_ckpt) forward(block_input_cache, 0, false);
        Matrix d_z2 = d_out;
        Matrix d_ln2 = use_moe ? moe.backward(d_out) : single_ffn.backward(d_out);
        d_z2 += ln2.backward(d_ln2);
        Matrix d_x = d_z2;
        d_x += ln1.backward(mha.backward(d_z2));
        return d_x;
    }

    double get_balance_loss() const { return use_moe ? moe.get_balance_loss() : 0.0; }

    void update_weights(Optimizer& opt) {
        ln1.update_weights(opt); mha.update_weights(opt);
        ln2.update_weights(opt);
        if (use_moe) moe.update_weights(opt); else single_ffn.update_weights(opt);
    }
    void zero_grad() {
        ln1.zero_grad(); mha.zero_grad(); ln2.zero_grad();
        if (use_moe) moe.zero_grad(); else single_ffn.zero_grad();
    }
    void clear_cache() { mha.clear_cache(); }
    void save(std::ostream& os) const {
        ln1.save(os); mha.save(os); ln2.save(os);
        if (use_moe) moe.save(os); else single_ffn.save(os);
    }
    void load(std::istream& is) {
        ln1.load(is); mha.load(is); ln2.load(is);
        if (use_moe) moe.load(is); else single_ffn.load(is);
    }
    size_t param_count() const {
        size_t t = ln1.param_count() + mha.param_count() + ln2.param_count();
        return t + (use_moe ? moe.param_count() : single_ffn.param_count());
    }
};

class Embedding {
    Matrix W, dW;
    ParamState sw;
    size_t vocab_size, dim;
    std::vector<int> ids_cache;
public:
    Embedding() : vocab_size(0), dim(0) {}
    Embedding(size_t vs, size_t d) : vocab_size(vs), dim(d) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 0.02);
        W = Matrix(vs, d); for (auto& v : W.data) v = dist(gen);
        sw.init(vs, d); dW = Matrix::zeros(vs, d);
    }
    Matrix forward(const std::vector<int>& ids) {
        ids_cache = ids;
        Matrix out(ids.size(), dim);
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = 0; j < dim; ++j) out(i, j) = W(static_cast<size_t>(ids[i]), j);
        return out;
    }
    void backward(const Matrix& d_out) {
        for (size_t i = 0; i < ids_cache.size(); ++i)
            for (size_t j = 0; j < dim; ++j) dW(static_cast<size_t>(ids_cache[i]), j) += d_out(i, j);
    }
    const Matrix& weight() const { return W; }
    void add_grad(const Matrix& g) { dW += g; }
    void update_weights(Optimizer& opt) { opt.update(W, dW, sw); }
    void zero_grad() { dW = Matrix::zeros(vocab_size, dim); }
    void save(std::ostream& os) const { W.serialize(os); sw.serialize(os); }
    void load(std::istream& is) { W = Matrix::deserialize(is); sw.deserialize(is); dW = Matrix::zeros(vocab_size, dim); }
    size_t param_count() const { return vocab_size * dim; }
};

struct GPTConfig {
    size_t vocab_size = 128;
    size_t d_model = 64;
    size_t num_heads = 4;
    size_t num_kv_heads = 4;
    size_t num_layers = 2;
    size_t d_ff = 256;
    size_t max_seq_len = 128;
    double lr = 3e-4;
    double grad_clip = 1.0;
    double weight_decay = 0.01;
    double min_lr = 1e-5;
    size_t warmup_steps = 1000;
    size_t total_steps = 1000000;
    size_t grad_accum_steps = 1;
    int top_k = 40;
    double top_p = 0.9;
    double repetition_penalty = 1.1;
    size_t num_experts = 1;
    size_t active_experts = 1;
    double moe_balance_coeff = 0.01;
    size_t window_size = 0;
    uint8_t grad_checkpoint = 0;

    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(this), sizeof(GPTConfig));
    }
    static GPTConfig deserialize(std::istream& is) {
        GPTConfig c; is.read(reinterpret_cast<char*>(&c), sizeof(GPTConfig)); return c;
    }
};

struct TrainState {
    uint64_t step = 0, total_tokens = 0;
    double loss_ema = 0.0;
    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&step), 8);
        os.write(reinterpret_cast<const char*>(&total_tokens), 8);
        os.write(reinterpret_cast<const char*>(&loss_ema), 8);
    }
    void deserialize(std::istream& is) {
        is.read(reinterpret_cast<char*>(&step), 8);
        is.read(reinterpret_cast<char*>(&total_tokens), 8);
        is.read(reinterpret_cast<char*>(&loss_ema), 8);
    }
};

class GPTModel {
    GPTConfig cfg;
    Embedding token_emb;
    std::vector<TransformerBlock> blocks;
    RMSNorm final_ln;
    Optimizer opt;
    LRScheduler lr_sched;
    Matrix final_ln_out_cache;
    std::vector<int> last_ids;

    Matrix clip_grad(const Matrix& g, double mn) {
        double n = 0.0;
        for (const auto& v : g.data) n += v * v;
        n = std::sqrt(n);
        if (n > mn) return g * (mn / n);
        return g;
    }

    Matrix softmax_rows(const Matrix& x) {
        Matrix r(x.rows, x.cols);
        for (size_t i = 0; i < x.rows; ++i) {
            double mx = x(i, 0);
            for (size_t j = 1; j < x.cols; ++j) if (x(i, j) > mx) mx = x(i, j);
            double s = 0.0;
            for (size_t j = 0; j < x.cols; ++j) { r(i, j) = std::exp(x(i, j) - mx); s += r(i, j); }
            for (size_t j = 0; j < x.cols; ++j) r(i, j) /= s;
        }
        return r;
    }

    int sample_token(const Matrix& logits, size_t row, double temp,
                      int top_k, double top_p, double rep_pen,
                      const std::vector<int>& recent, std::mt19937& rng) {
        size_t V = cfg.vocab_size;
        std::vector<std::pair<double, int>> sc(V);
        for (size_t j = 0; j < V; ++j) {
            double s = logits(row, j);
            for (size_t r = (recent.size() > cfg.max_seq_len ? recent.size() - cfg.max_seq_len : 0);
                 r < recent.size(); ++r)
                if (static_cast<size_t>(recent[r]) == j) {
                    s = (s > 0) ? s / rep_pen : s * rep_pen; break;
                }
            sc[j] = {s / temp, static_cast<int>(j)};
        }
        std::sort(sc.begin(), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        size_t k = std::min(static_cast<size_t>(top_k), V);
        double mx = sc[0].first, sum = 0.0;
        std::vector<double> probs(k);
        for (size_t i = 0; i < k; ++i) { probs[i] = std::exp(sc[i].first - mx); sum += probs[i]; }
        for (size_t i = 0; i < k; ++i) probs[i] /= sum;
        double cum = 0.0; size_t cutoff = k;
        for (size_t i = 0; i < k; ++i) { cum += probs[i]; if (cum >= top_p) { cutoff = i + 1; break; } }
        if (cutoff < k) { sum = 0.0; for (size_t i = 0; i < cutoff; ++i) sum += probs[i]; for (size_t i = 0; i < cutoff; ++i) probs[i] /= sum; }
        std::vector<double> w(probs.begin(), probs.begin() + static_cast<long>(cutoff));
        std::discrete_distribution<int> dist(w.begin(), w.end());
        return sc[static_cast<size_t>(dist(rng))].second;
    }

public:
    GPTModel() = default;

    explicit GPTModel(GPTConfig c) : cfg(c) {
        token_emb = Embedding(c.vocab_size, c.d_model);
        for (size_t i = 0; i < c.num_layers; ++i)
            blocks.emplace_back(c.d_model, c.num_heads, c.num_kv_heads,
                                 c.d_ff, c.max_seq_len, c.window_size,
                                 c.num_experts, c.active_experts,
                                 c.moe_balance_coeff, c.grad_checkpoint != 0);
        final_ln = RMSNorm(c.d_model);
        opt = Optimizer(OptimizerConfig{
            .type = OptimizerType::AdamW, .lr = c.lr,
            .beta1 = 0.9, .beta2 = 0.98, .epsilon = 1e-9, .weight_decay = c.weight_decay
        });
        lr_sched = LRScheduler(c.lr, c.min_lr, c.warmup_steps, c.total_steps);
    }

    Matrix forward(const std::vector<int>& ids, size_t sp = 0, bool uc = false) {
        last_ids = ids;
        Matrix x = token_emb.forward(ids);
        for (auto& b : blocks) x = b.forward(x, sp, uc);
        final_ln_out_cache = final_ln.forward(x);
        return final_ln_out_cache * token_emb.weight().T();
    }

    void zero_grad() {
        final_ln.zero_grad();
        for (auto& b : blocks) b.zero_grad();
        token_emb.zero_grad();
    }

    double compute_loss_backward(const std::vector<int>& input_ids, const std::vector<int>& target_ids) {
        clear_cache();
        Matrix logits = forward(input_ids, 0, false);
        size_t seq = logits.rows;
        Matrix probs = softmax_rows(logits);
        double loss_val = 0.0;
        for (size_t i = 0; i < seq; ++i)
            loss_val -= std::log(std::max(probs(i, static_cast<size_t>(target_ids[i])), 1e-10));
        loss_val /= static_cast<double>(seq);

        double bal_loss = 0.0;
        for (const auto& b : blocks) bal_loss += b.get_balance_loss();

        Matrix d_logits(seq, cfg.vocab_size);
        for (size_t i = 0; i < seq; ++i) {
            for (size_t j = 0; j < cfg.vocab_size; ++j) d_logits(i, j) = probs(i, j);
            d_logits(i, static_cast<size_t>(target_ids[i])) -= 1.0;
            for (size_t j = 0; j < cfg.vocab_size; ++j) d_logits(i, j) /= static_cast<double>(seq);
        }
        d_logits = clip_grad(d_logits, cfg.grad_clip);
        Matrix d_fln = d_logits * token_emb.weight();
        token_emb.add_grad(d_logits.T() * final_ln_out_cache);
        Matrix d_x = final_ln.backward(d_fln);
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
            d_x = blocks[static_cast<size_t>(i)].backward(d_x);
        token_emb.backward(d_x);
        return loss_val + bal_loss;
    }

    void apply_update(size_t accum = 1) {
        opt.set_lr(lr_sched.step());
        (void)accum;
        final_ln.update_weights(opt);
        for (auto& b : blocks) b.update_weights(opt);
        token_emb.update_weights(opt);
    }

    double train_step(const std::vector<int>& in, const std::vector<int>& tgt) {
        zero_grad(); double l = compute_loss_backward(in, tgt); apply_update(1); return l;
    }

    double current_lr() const { return lr_sched.get_lr(); }

    std::vector<int> generate(const std::vector<int>& prompt, size_t n, double temp = 0.8) {
        clear_cache();
        std::vector<int> res = prompt;
        std::mt19937 rng(std::random_device{}());
        Matrix logits = forward(prompt, 0, true);
        int tok = sample_token(logits, logits.rows - 1, temp, cfg.top_k, cfg.top_p, cfg.repetition_penalty, res, rng);
        res.push_back(tok);
        for (size_t t = 1; t < n; ++t) {
            size_t pos = res.size() - 1;
            if (pos >= cfg.max_seq_len) break;
            logits = forward({res.back()}, pos, true);
            tok = sample_token(logits, 0, temp, cfg.top_k, cfg.top_p, cfg.repetition_penalty, res, rng);
            res.push_back(tok);
        }
        return res;
    }

    void clear_cache() { for (auto& b : blocks) b.clear_cache(); }

    bool save_checkpoint(const std::string& path, const TrainState& ts) const {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        f.write("XGPT", 4);
        uint32_t v = 5; f.write(reinterpret_cast<const char*>(&v), 4);
        cfg.serialize(f); ts.serialize(f); lr_sched.serialize(f);
        token_emb.save(f);
        uint32_t nl = static_cast<uint32_t>(blocks.size());
        f.write(reinterpret_cast<const char*>(&nl), 4);
        for (const auto& b : blocks) b.save(f);
        final_ln.save(f); f.close(); return f.good();
    }

    bool load_checkpoint(const std::string& path, TrainState& ts) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        char magic[4]; f.read(magic, 4);
        if (magic[0] != 'X' || magic[1] != 'G' || magic[2] != 'P' || magic[3] != 'T') return false;
        uint32_t v; f.read(reinterpret_cast<char*>(&v), 4);
        if (v < 5) return false;
        cfg = GPTConfig::deserialize(f); ts.deserialize(f); lr_sched.deserialize(f);
        token_emb = Embedding(cfg.vocab_size, cfg.d_model); token_emb.load(f);
        uint32_t nl; f.read(reinterpret_cast<char*>(&nl), 4);
        blocks.clear();
        for (uint32_t i = 0; i < nl; ++i) {
            blocks.emplace_back(cfg.d_model, cfg.num_heads, cfg.num_kv_heads,
                                 cfg.d_ff, cfg.max_seq_len, cfg.window_size,
                                 cfg.num_experts, cfg.active_experts,
                                 cfg.moe_balance_coeff, cfg.grad_checkpoint != 0);
            blocks.back().load(f);
        }
        final_ln = RMSNorm(cfg.d_model); final_ln.load(f);
        opt = Optimizer(OptimizerConfig{
            .type = OptimizerType::AdamW, .lr = cfg.lr,
            .beta1 = 0.9, .beta2 = 0.98, .epsilon = 1e-9, .weight_decay = cfg.weight_decay
        });
        f.close(); return f.good();
    }

    size_t total_params() const {
        size_t t = token_emb.param_count() + final_ln.param_count();
        for (const auto& b : blocks) t += b.param_count();
        return t;
    }

    void summary() const {
        std::string gqa = (cfg.num_kv_heads < cfg.num_heads) ?
            "GQA(" + std::to_string(cfg.num_kv_heads) + ")" : "MHA";
        std::string ffn = (cfg.num_experts > 1) ?
            "MoE(" + std::to_string(cfg.num_experts) + "E x" +
            std::to_string(cfg.active_experts) + "A SwiGLU)" : "SwiGLU";
        std::string win = (cfg.window_size > 0) ?
            std::to_string(cfg.window_size) : "Full";

        std::cout << util::color::BOLD << util::color::CYAN << R"(
    ╔═══════════════════════════════════════════════╗
    ║  XasmAI GPT — Mixtral-Class Architecture      ║
    ╠═══════════════════════════════════════════════╣
)" << util::color::RESET;
        std::cout << util::color::WHITE
                  << "    ║  Vocab:       " << std::setw(8) << cfg.vocab_size << "                      ║\n"
                  << "    ║  d_model:     " << std::setw(8) << cfg.d_model << "                      ║\n"
                  << "    ║  Heads:       " << std::setw(8) << cfg.num_heads << "                      ║\n"
                  << "    ║  KV Heads:    " << std::setw(8) << cfg.num_kv_heads << "                      ║\n"
                  << "    ║  Layers:      " << std::setw(8) << cfg.num_layers << "                      ║\n"
                  << "    ║  d_ff:        " << std::setw(8) << cfg.d_ff << "                      ║\n"
                  << "    ║  Max Seq:     " << std::setw(8) << cfg.max_seq_len << "                      ║\n"
                  << util::color::GREEN
                  << "    ║  Attention:   " << std::setw(8) << gqa << "  [FlashAttn]         ║\n"
                  << "    ║  FFN:         " << std::setw(25) << ffn << "  ║\n"
                  << "    ║  Norm:        " << " RMSNorm" << "                      ║\n"
                  << "    ║  Pos Embed:   " << "    RoPE" << "                      ║\n"
                  << "    ║  Optimizer:   " << "   AdamW" << "                      ║\n"
                  << "    ║  W. Tying:    " << "     YES" << "                      ║\n"
                  << util::color::YELLOW
                  << "    ║  Window:      " << std::setw(8) << win << "                      ║\n"
                  << "    ║  Top-k:       " << std::setw(8) << cfg.top_k << "                      ║\n"
                  << "    ║  Top-p:       " << std::setw(8) << std::fixed << std::setprecision(2) << cfg.top_p << "                      ║\n"
                  << "    ║  Rep.Pen:     " << std::setw(8) << cfg.repetition_penalty << "                      ║\n"
                  << "    ║  GradCkpt:    " << std::setw(8) << (cfg.grad_checkpoint ? "YES" : "NO") << "                      ║\n"
                  << util::color::CYAN
                  << "    ║  Parameters:  " << std::setw(8) << total_params() << "  (tied)              ║\n"
                  << "    ╚═══════════════════════════════════════════════╝\n\n"
                  << util::color::RESET;
    }

    const GPTConfig& config() const { return cfg; }
    const LRScheduler& scheduler() const { return lr_sched; }
};

} // namespace xasm
