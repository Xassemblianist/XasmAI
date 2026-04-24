#pragma once

#include "matrix.hpp"
#include "optimizer.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

namespace xasm {

class RoPE {
    std::vector<double> cos_tab, sin_tab;
    size_t half_dim, max_len, dim;

public:
    RoPE() : half_dim(0), max_len(0), dim(0) {}

    RoPE(size_t d, size_t ml) : half_dim(d / 2), max_len(ml), dim(d) {
        cos_tab.resize(ml * half_dim);
        sin_tab.resize(ml * half_dim);
        for (size_t pos = 0; pos < ml; ++pos)
            for (size_t i = 0; i < half_dim; ++i) {
                double theta = static_cast<double>(pos) /
                    std::pow(10000.0, 2.0 * static_cast<double>(i) / static_cast<double>(d));
                cos_tab[pos * half_dim + i] = std::cos(theta);
                sin_tab[pos * half_dim + i] = std::sin(theta);
            }
    }

    Matrix forward(const Matrix& x, size_t start_pos) const {
        Matrix out(x.rows, x.cols);
        for (size_t p = 0; p < x.rows; ++p) {
            size_t abs_pos = start_pos + p;
            if (abs_pos >= max_len) abs_pos = max_len - 1;
            for (size_t i = 0; i < half_dim && 2 * i + 1 < x.cols; ++i) {
                double c = cos_tab[abs_pos * half_dim + i];
                double s = sin_tab[abs_pos * half_dim + i];
                out(p, 2 * i)     = x(p, 2 * i) * c - x(p, 2 * i + 1) * s;
                out(p, 2 * i + 1) = x(p, 2 * i) * s + x(p, 2 * i + 1) * c;
            }
            if (dim % 2 == 1) out(p, dim - 1) = x(p, dim - 1);
        }
        return out;
    }

    Matrix backward(const Matrix& d_out, size_t start_pos) const {
        Matrix dx(d_out.rows, d_out.cols);
        for (size_t p = 0; p < d_out.rows; ++p) {
            size_t abs_pos = start_pos + p;
            if (abs_pos >= max_len) abs_pos = max_len - 1;
            for (size_t i = 0; i < half_dim && 2 * i + 1 < d_out.cols; ++i) {
                double c = cos_tab[abs_pos * half_dim + i];
                double s = sin_tab[abs_pos * half_dim + i];
                dx(p, 2 * i)     =  d_out(p, 2 * i) * c + d_out(p, 2 * i + 1) * s;
                dx(p, 2 * i + 1) = -d_out(p, 2 * i) * s + d_out(p, 2 * i + 1) * c;
            }
            if (dim % 2 == 1) dx(p, dim - 1) = d_out(p, dim - 1);
        }
        return dx;
    }
};

class MultiHeadAttention {
    size_t d_model, num_heads, num_kv_heads, d_k;
    size_t kv_dim, group_size, window_size_;

    Matrix Wq, Wk, Wv, Wo;
    Matrix dWq, dWk, dWv, dWo;
    ParamState sWq, sWk, sWv, sWo;
    RoPE rope;
    Matrix input_cache;

    struct QHeadCache { Matrix Q_rot, lse; };
    struct KVHeadCache { Matrix K_rot, V; };
    std::vector<QHeadCache> q_caches;
    std::vector<KVHeadCache> kv_train_caches;
    Matrix concat_cache;

    struct InferKVCache { Matrix k, v; };
    std::vector<InferKVCache> kv_caches;

    static void append_rows(Matrix& dst, const Matrix& src) {
        if (dst.rows == 0) { dst = src; return; }
        Matrix n(dst.rows + src.rows, dst.cols);
        for (size_t i = 0; i < dst.rows; ++i)
            for (size_t j = 0; j < dst.cols; ++j) n(i, j) = dst(i, j);
        for (size_t i = 0; i < src.rows; ++i)
            for (size_t j = 0; j < src.cols; ++j) n(dst.rows + i, j) = src(i, j);
        dst = std::move(n);
    }

    static Matrix trim_front(const Matrix& m, size_t n) {
        if (n >= m.rows) return Matrix();
        Matrix r(m.rows - n, m.cols);
        for (size_t i = 0; i < r.rows; ++i)
            for (size_t j = 0; j < r.cols; ++j) r(i, j) = m(n + i, j);
        return r;
    }

    Matrix extract_head(const Matrix& proj, size_t h, size_t head_dim, size_t hdim) const {
        Matrix out(proj.rows, hdim);
        size_t off = h * hdim;
        for (size_t i = 0; i < proj.rows; ++i)
            for (size_t j = 0; j < hdim; ++j) out(i, j) = proj(i, off + j);
        (void)head_dim;
        return out;
    }

    void insert_head(Matrix& dst, const Matrix& src, size_t h, size_t hdim) const {
        size_t off = h * hdim;
        for (size_t i = 0; i < src.rows; ++i)
            for (size_t j = 0; j < hdim; ++j) dst(i, off + j) = src(i, j);
    }

    Matrix flash_forward(const Matrix& Q, const Matrix& K, const Matrix& V,
                          double scale, size_t start_pos, Matrix& lse_out) const {
        size_t sq = Q.rows, skv = K.rows;
        Matrix output(sq, d_k, 0.0);
        lse_out = Matrix(sq, 1);

        for (size_t i = 0; i < sq; ++i) {
            double rmax = -1e30, rsum = 0.0;
            size_t abs_i = start_pos + i;
            size_t j_end = std::min(abs_i + 1, skv);
            size_t j_start = 0;
            if (window_size_ > 0 && abs_i >= window_size_)
                j_start = abs_i - window_size_ + 1;
            if (j_start >= skv) j_start = 0;

            for (size_t j = j_start; j < j_end; ++j) {
                double s = 0.0;
                for (size_t d = 0; d < d_k; ++d) s += Q(i, d) * K(j, d);
                s *= scale;
                double old_max = rmax;
                rmax = std::max(rmax, s);
                double corr = std::exp(old_max - rmax);
                rsum = rsum * corr + std::exp(s - rmax);
                for (size_t d = 0; d < d_k; ++d)
                    output(i, d) = output(i, d) * corr + std::exp(s - rmax) * V(j, d);
            }
            if (rsum > 0.0) {
                double inv = 1.0 / rsum;
                for (size_t d = 0; d < d_k; ++d) output(i, d) *= inv;
            }
            lse_out(i, 0) = std::log(std::max(rsum, 1e-30)) + rmax;
        }
        return output;
    }

    void flash_backward(const Matrix& d_out_h, const Matrix& Q_rot,
                          const Matrix& K_rot, const Matrix& V,
                          const Matrix& lse, double scale,
                          Matrix& dQ_rot, Matrix& dK_rot, Matrix& dV) const {
        size_t sq = Q_rot.rows, skv = K_rot.rows;
        for (size_t i = 0; i < sq; ++i) {
            size_t j_end = std::min(i + 1, skv);
            size_t j_start = 0;
            if (window_size_ > 0 && i >= window_size_)
                j_start = i - window_size_ + 1;

            std::vector<double> attn(j_end > j_start ? j_end - j_start : 0);
            for (size_t j = j_start; j < j_end; ++j) {
                double s = 0.0;
                for (size_t d = 0; d < d_k; ++d) s += Q_rot(i, d) * K_rot(j, d);
                attn[j - j_start] = std::exp(s * scale - lse(i, 0));
            }

            for (size_t j = j_start; j < j_end; ++j)
                for (size_t d = 0; d < d_k; ++d)
                    dV(j, d) += attn[j - j_start] * d_out_h(i, d);

            std::vector<double> d_attn(attn.size());
            for (size_t j = j_start; j < j_end; ++j) {
                double da = 0.0;
                for (size_t d = 0; d < d_k; ++d) da += d_out_h(i, d) * V(j, d);
                d_attn[j - j_start] = da;
            }

            double sum_ad = 0.0;
            for (size_t idx = 0; idx < attn.size(); ++idx)
                sum_ad += attn[idx] * d_attn[idx];

            for (size_t j = j_start; j < j_end; ++j) {
                double ds = attn[j - j_start] * (d_attn[j - j_start] - sum_ad) * scale;
                for (size_t d = 0; d < d_k; ++d) {
                    dQ_rot(i, d) += ds * K_rot(j, d);
                    dK_rot(j, d) += ds * Q_rot(i, d);
                }
            }
        }
    }

public:
    MultiHeadAttention() : d_model(0), num_heads(0), num_kv_heads(0), d_k(0),
                            kv_dim(0), group_size(1), window_size_(0) {}

    MultiHeadAttention(size_t dm, size_t nh, size_t nkv, size_t max_seq_len, size_t ws = 0)
        : d_model(dm), num_heads(nh), num_kv_heads(nkv),
          d_k(dm / nh), kv_dim(nkv * (dm / nh)), group_size(nh / nkv),
          window_size_(ws), rope(dm / nh, max_seq_len) {
        double s = std::sqrt(2.0 / static_cast<double>(dm));
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, s);
        auto mk = [&](size_t r, size_t c) {
            Matrix m(r, c); for (auto& v : m.data) v = dist(gen); return m; };
        Wq = mk(dm, dm); Wk = mk(dm, kv_dim);
        Wv = mk(dm, kv_dim); Wo = mk(dm, dm);
        sWq.init(dm, dm); sWk.init(dm, kv_dim);
        sWv.init(dm, kv_dim); sWo.init(dm, dm);
        q_caches.resize(nh); kv_train_caches.resize(nkv); kv_caches.resize(nkv);
        zero_grad();
    }

    Matrix forward(const Matrix& x, size_t start_pos = 0, bool use_cache = false) {
        input_cache = x;
        size_t seq = x.rows;
        double scale = 1.0 / std::sqrt(static_cast<double>(d_k));
        Matrix Q_proj = x * Wq, K_proj = x * Wk, V_proj = x * Wv;

        for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            Matrix Kh = extract_head(K_proj, kv_h, 0, d_k);
            Matrix Vh = extract_head(V_proj, kv_h, 0, d_k);
            Matrix Kr = rope.forward(Kh, start_pos);
            if (use_cache) {
                append_rows(kv_caches[kv_h].k, Kr);
                append_rows(kv_caches[kv_h].v, Vh);
                if (window_size_ > 0 && kv_caches[kv_h].k.rows > window_size_) {
                    size_t excess = kv_caches[kv_h].k.rows - window_size_;
                    kv_caches[kv_h].k = trim_front(kv_caches[kv_h].k, excess);
                    kv_caches[kv_h].v = trim_front(kv_caches[kv_h].v, excess);
                }
            } else {
                kv_train_caches[kv_h].K_rot = Kr;
                kv_train_caches[kv_h].V = Vh;
            }
        }

        concat_cache = Matrix(seq, d_model);
        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / group_size;
            Matrix Qh = extract_head(Q_proj, h, 0, d_k);
            Matrix Qr = rope.forward(Qh, start_pos);
            Matrix K_attn = use_cache ? kv_caches[kv_h].k : kv_train_caches[kv_h].K_rot;
            Matrix V_attn = use_cache ? kv_caches[kv_h].v : kv_train_caches[kv_h].V;
            Matrix lse;
            Matrix head_out = flash_forward(Qr, K_attn, V_attn, scale, start_pos, lse);
            q_caches[h].Q_rot = Qr;
            q_caches[h].lse = lse;
            insert_head(concat_cache, head_out, h, d_k);
        }
        return concat_cache * Wo;
    }

    Matrix backward(const Matrix& d_output) {
        dWo += concat_cache.T() * d_output;
        Matrix d_concat = d_output * Wo.T();
        size_t seq = input_cache.rows;
        double scale = 1.0 / std::sqrt(static_cast<double>(d_k));
        Matrix dQ_proj = Matrix::zeros(seq, d_model);
        std::vector<Matrix> dK_rot_a(num_kv_heads, Matrix::zeros(seq, d_k));
        std::vector<Matrix> dV_a(num_kv_heads, Matrix::zeros(seq, d_k));

        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / group_size;
            Matrix d_head = extract_head(d_concat, h, 0, d_k);
            Matrix dQr = Matrix::zeros(seq, d_k);
            Matrix dKr = Matrix::zeros(seq, d_k);
            Matrix dVh = Matrix::zeros(seq, d_k);
            flash_backward(d_head, q_caches[h].Q_rot, kv_train_caches[kv_h].K_rot,
                            kv_train_caches[kv_h].V, q_caches[h].lse, scale, dQr, dKr, dVh);
            insert_head(dQ_proj, rope.backward(dQr, 0), h, d_k);
            dK_rot_a[kv_h] += dKr;
            dV_a[kv_h] += dVh;
        }
        Matrix dK_proj = Matrix::zeros(seq, kv_dim);
        Matrix dV_proj = Matrix::zeros(seq, kv_dim);
        for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            insert_head(dK_proj, rope.backward(dK_rot_a[kv_h], 0), kv_h, d_k);
            insert_head(dV_proj, dV_a[kv_h], kv_h, d_k);
        }
        dWq += input_cache.T() * dQ_proj;
        dWk += input_cache.T() * dK_proj;
        dWv += input_cache.T() * dV_proj;
        return dQ_proj * Wq.T() + dK_proj * Wk.T() + dV_proj * Wv.T();
    }

    void update_weights(Optimizer& opt) {
        opt.update(Wq, dWq, sWq); opt.update(Wk, dWk, sWk);
        opt.update(Wv, dWv, sWv); opt.update(Wo, dWo, sWo);
    }
    void zero_grad() {
        dWq = Matrix::zeros(d_model, d_model); dWk = Matrix::zeros(d_model, kv_dim);
        dWv = Matrix::zeros(d_model, kv_dim); dWo = Matrix::zeros(d_model, d_model);
    }
    void clear_cache() { for (auto& c : kv_caches) { c.k = Matrix(); c.v = Matrix(); } }
    void save(std::ostream& os) const {
        Wq.serialize(os); Wk.serialize(os); Wv.serialize(os); Wo.serialize(os);
        sWq.serialize(os); sWk.serialize(os); sWv.serialize(os); sWo.serialize(os);
    }
    void load(std::istream& is) {
        Wq = Matrix::deserialize(is); Wk = Matrix::deserialize(is);
        Wv = Matrix::deserialize(is); Wo = Matrix::deserialize(is);
        sWq.deserialize(is); sWk.deserialize(is); sWv.deserialize(is); sWo.deserialize(is);
        zero_grad(); q_caches.resize(num_heads);
        kv_train_caches.resize(num_kv_heads); kv_caches.resize(num_kv_heads);
    }
    size_t param_count() const { return d_model * d_model + 2 * d_model * kv_dim + d_model * d_model; }
};

} // namespace xasm
