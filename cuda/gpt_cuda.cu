#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <set>
#include <map>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace fs = std::filesystem;

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x);if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d err=%d\n",__FILE__,__LINE__,s);exit(1);}}while(0)

static volatile sig_atomic_t g_stop = 0;
static void sig_handler(int) { g_stop = 1; }

inline int DV(int a, int b) { return (a + b - 1) / b; }

float* ga(size_t n) { float* p; CK(cudaMalloc(&p, n * sizeof(float))); CK(cudaMemset(p, 0, n * sizeof(float))); return p; }
void gf(float* p) { if (p) CK(cudaFree(p)); }
void h2d(float* d, const float* h, size_t n) { CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice)); }
void d2h(float* h, const float* d, size_t n) { CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost)); }
void d2d(float* d, const float* s, size_t n) { CK(cudaMemcpy(d, s, n * sizeof(float), cudaMemcpyDeviceToDevice)); }
void gz(float* p, size_t n) { CK(cudaMemset(p, 0, n * sizeof(float))); }

void gemm_nn(cublasHandle_t h, float* A, float* B, float* C, int M, int K, int N) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, B, N, A, K, &b, C, N));
}
void gemm_nn_acc(cublasHandle_t h, float* A, float* B, float* C, int M, int K, int N) {
    float a = 1.0f, b = 1.0f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, B, N, A, K, &b, C, N));
}
void gemm_nt(cublasHandle_t h, float* A, float* B, float* C, int M, int K, int N) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, B, K, A, K, &b, C, N));
}
void gemm_tn(cublasHandle_t h, float* A, float* B, float* C, int M, int K, int N) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &a, B, N, A, K, &b, C, N));
}
void gemm_tn_acc(cublasHandle_t h, float* A, float* B, float* C, int M, int K, int N) {
    float a = 1.0f, b = 1.0f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &a, B, N, A, K, &b, C, N));
}

void bgemm_nt(cublasHandle_t h, float* A, float* B, float* C,
              int M, int K, int N, int batch, long long sA, long long sB, long long sC) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &a, B, K, sB, A, K, sA, &b, C, N, sC, batch));
}
void bgemm_nn(cublasHandle_t h, float* A, float* B, float* C,
              int M, int K, int N, int batch, long long sA, long long sB, long long sC) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &a, B, N, sB, A, K, sA, &b, C, N, sC, batch));
}
void bgemm_tn(cublasHandle_t h, float* A, float* B, float* C,
              int M, int K, int N, int batch, long long sA, long long sB, long long sC) {
    float a = 1.0f, b = 0.0f;
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M,
        &a, B, N, sB, A, K, sA, &b, C, N, sC, batch));
}

void bgemm_nt_s(cublasHandle_t h, float* A, float* B, float* C,
                int M, int K, int N, int batch, long long sA, long long sB, long long sC, float scale) {
    float b = 0.0f;
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &scale, B, K, sB, A, K, sA, &b, C, N, sC, batch));
}

__global__ void k_add(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_embed(float* out, const float* W, const int* ids, int seq, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq * dim) return;
    out[i] = W[ids[i / dim] * dim + i % dim];
}

__global__ void k_embed_bwd(float* dW, const float* dout, const int* ids, int seq, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq * dim) return;
    atomicAdd(&dW[ids[i / dim] * dim + i % dim], dout[i]);
}

__global__ void k_rmsnorm(float* out, float* rms, float* xn, const float* x, const float* g, int seq, int dim) {
    extern __shared__ float s[];
    int row = blockIdx.x; if (row >= seq) return;
    int t = threadIdx.x;
    const float* xr = x + row * dim;
    float lsq = 0.0f;
    for (int j = t; j < dim; j += blockDim.x) lsq += xr[j] * xr[j];
    s[t] = lsq; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    float r = sqrtf(s[0] / (float)dim + 1e-6f);
    if (t == 0) rms[row] = r;
    float inv = 1.0f / r;
    for (int j = t; j < dim; j += blockDim.x) {
        float v = xr[j] * inv;
        xn[row * dim + j] = v;
        out[row * dim + j] = g[j] * v;
    }
}

__global__ void k_rmsnorm_bwd(float* dx, float* dg, const float* dout, const float* xn,
                               const float* rms_v, const float* g, int seq, int dim) {
    extern __shared__ float s[];
    int row = blockIdx.x; if (row >= seq) return;
    int t = threadIdx.x;
    const float* dr = dout + row * dim;
    const float* nr = xn + row * dim;
    float N = (float)dim, rv = rms_v[row];
    float ld = 0.0f;
    for (int j = t; j < dim; j += blockDim.x) ld += dr[j] * g[j] * nr[j];
    s[t] = ld; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    float sd = s[0];
    for (int j = t; j < dim; j += blockDim.x) {
        dx[row * dim + j] = (dr[j] * g[j] - nr[j] * sd / N) / rv;
        atomicAdd(&dg[j], dr[j] * nr[j]);
    }
}

__global__ void k_reshape_h(float* out, const float* in, int seq, int nh, int dk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * seq * dk;
    if (i >= total) return;
    int h = i / (seq * dk), rem = i % (seq * dk), s = rem / dk, d = rem % dk;
    out[i] = in[s * nh * dk + h * dk + d];
}

__global__ void k_unreshape_h(float* out, const float* in, int seq, int nh, int dk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * seq * dk;
    if (i >= total) return;
    int h = i / (seq * dk), rem = i % (seq * dk), s = rem / dk, d = rem % dk;
    out[s * nh * dk + h * dk + d] = in[i];
}

__global__ void k_rope(float* out, const float* x, const float* ct, const float* st,
                        int nh, int seq, int dk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hd = dk / 2, total = nh * seq * hd;
    if (i >= total) return;
    int h = i / (seq * hd), rem = i % (seq * hd), s = rem / hd, p = rem % hd;
    int base = h * seq * dk + s * dk;
    float c = ct[s * hd + p], sn = st[s * hd + p];
    float x0 = x[base + 2 * p], x1 = x[base + 2 * p + 1];
    out[base + 2 * p] = x0 * c - x1 * sn;
    out[base + 2 * p + 1] = x0 * sn + x1 * c;
}

__global__ void k_rope_bwd(float* dx, const float* dout, const float* ct, const float* st,
                            int nh, int seq, int dk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hd = dk / 2, total = nh * seq * hd;
    if (i >= total) return;
    int h = i / (seq * hd), rem = i % (seq * hd), s = rem / hd, p = rem % hd;
    int base = h * seq * dk + s * dk;
    float c = ct[s * hd + p], sn = st[s * hd + p];
    float d0 = dout[base + 2 * p], d1 = dout[base + 2 * p + 1];
    dx[base + 2 * p] = d0 * c + d1 * sn;
    dx[base + 2 * p + 1] = -d0 * sn + d1 * c;
}

__global__ void k_causal_softmax(float* probs, const float* scores, int nh, int seq) {
    extern __shared__ float s[];
    int h = blockIdx.y, row = blockIdx.x;
    if (row >= seq) return;
    int t = threadIdx.x;
    int off = h * seq * seq + row * seq;
    float lm = -1e30f;
    for (int j = t; j <= row; j += blockDim.x) lm = fmaxf(lm, scores[off + j]);
    s[t] = lm; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] = fmaxf(s[t], s[t + k]); __syncthreads(); }
    float mx = s[0];
    float ls = 0.0f;
    for (int j = t; j <= row; j += blockDim.x) ls += expf(scores[off + j] - mx);
    s[t] = ls; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    float sm = s[0];
    for (int j = t; j < seq; j += blockDim.x)
        probs[off + j] = (j <= row) ? expf(scores[off + j] - mx) / sm : 0.0f;
}

__global__ void k_softmax_bwd(float* ds, const float* dp, const float* p, int nh, int seq) {
    extern __shared__ float s[];
    int h = blockIdx.y, row = blockIdx.x;
    if (row >= seq) return;
    int t = threadIdx.x;
    int off = h * seq * seq + row * seq;
    float ld = 0.0f;
    for (int j = t; j <= row; j += blockDim.x) ld += p[off + j] * dp[off + j];
    s[t] = ld; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    float dot = s[0];
    for (int j = t; j < seq; j += blockDim.x)
        ds[off + j] = (j <= row) ? p[off + j] * (dp[off + j] - dot) : 0.0f;
}

__global__ void k_silu_mul(float* hidden, float* ga_c, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], sig = 1.0f / (1.0f + expf(-g)), sv = g * sig;
    ga_c[i] = sv;
    hidden[i] = sv * up[i];
}

__global__ void k_silu_mul_bwd(float* dg, float* du, const float* dh,
                                const float* gate, const float* up, const float* ga_c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], sig = 1.0f / (1.0f + expf(-g));
    du[i] = dh[i] * ga_c[i];
    dg[i] = dh[i] * up[i] * sig * (1.0f + g * (1.0f - sig));
}

__global__ void k_ce_softmax(float* dl, float* loss_out, const float* logits,
                              const int* tgt, int seq, int V) {
    extern __shared__ float s[];
    int row = blockIdx.x; if (row >= seq) return;
    int t = threadIdx.x;
    const float* lr = logits + row * V;
    float* dr = dl + row * V;
    float lm = -1e30f;
    for (int j = t; j < V; j += blockDim.x) lm = fmaxf(lm, lr[j]);
    s[t] = lm; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] = fmaxf(s[t], s[t + k]); __syncthreads(); }
    float mx = s[0];
    float ls = 0.0f;
    for (int j = t; j < V; j += blockDim.x) ls += expf(lr[j] - mx);
    s[t] = ls; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    float sm = s[0];
    int tg = tgt[row];
    for (int j = t; j < V; j += blockDim.x) {
        float p = expf(lr[j] - mx) / sm;
        dr[j] = (p - (j == tg ? 1.0f : 0.0f)) / (float)seq;
    }
    if (t == 0) {
        float p = expf(lr[tg] - mx) / sm;
        atomicAdd(loss_out, -logf(fmaxf(p, 1e-10f)) / (float)seq);
    }
}

__global__ void k_sqsum(float* out, const float* x, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    float v = (i < n) ? x[i] * x[i] : 0.0f;
    s[t] = v; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (t < k) s[t] += s[t + k]; __syncthreads(); }
    if (t == 0) atomicAdd(out, s[0]);
}

__global__ void k_adamw(float* w, float* m, float* v, const float* g,
                         float lr, float b1, float b2, float eps, float wd,
                         float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    float mh = m[i] / (1.0f - bc1), vh = v[i] / (1.0f - bc2);
    w[i] = w[i] * (1.0f - lr * wd) - lr * mh / (sqrtf(vh) + eps);
}

struct P {
    float *w, *g, *m, *v;
    int sz;
    void init(int n) {
        sz = n; w = ga(n); g = ga(n); m = ga(n); v = ga(n);
    }
    void rand_fill(float std_dev) {
        std::vector<float> h(sz);
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> d(0.0f, std_dev);
        for (auto& x : h) x = d(rng);
        h2d(w, h.data(), sz);
    }
    void fill_ones() {
        std::vector<float> h(sz, 1.0f);
        h2d(w, h.data(), sz);
    }
    void zero_grad() { gz(g, sz); }
    void update(float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
        k_adamw<<<DV(sz, 256), 256>>>(w, m, v, g, lr, b1, b2, eps, wd, bc1, bc2, sz);
    }
    void save(FILE* f) {
        std::vector<float> h(sz); d2h(h.data(), w, sz); fwrite(h.data(), 4, sz, f);
        d2h(h.data(), m, sz); fwrite(h.data(), 4, sz, f);
        d2h(h.data(), v, sz); fwrite(h.data(), 4, sz, f);
    }
    void load(FILE* f) {
        std::vector<float> h(sz); fread(h.data(), 4, sz, f); h2d(w, h.data(), sz);
        fread(h.data(), 4, sz, f); h2d(m, h.data(), sz);
        fread(h.data(), 4, sz, f); h2d(v, h.data(), sz);
    }
    void release() { gf(w); gf(g); gf(m); gf(v); }
};

struct Config {
    int vocab, dim, heads, layers, dff, max_seq, dk;
};

struct Layer {
    P g1, Wq, Wk, Wv, Wo, g2, Wg, Wu, Wd;
    float *ln1_out, *xn1, *rms1;
    float *q_heads, *k_heads, *v_heads;
    float *q_rope, *k_rope;
    float *scores, *probs, *attn_out, *concat;
    float *residual;
    float *ln2_out, *xn2, *rms2;
    float *gate_pre, *up_pre, *gate_act, *hidden;
};

struct Workspace {
    // forward scratch (avoid aliasing x_out with buf1/buf2)
    float *fwd_qvp, *fwd_kp, *fwd_proj, *fwd_ffn_out;
    // backward scratch
    float *tmp_dh;
    float *d_gate, *d_up;
    float *d_ln2, *d_ln2_2;
    float *d_res1_ln2;
    float *d_concat;
    float *d_attn_out;
    float *d_probs, *d_scores;
    float *d_V, *d_Q_rope, *d_K_rope, *d_Q_heads, *d_K_heads;
    float *d_qp, *d_kp, *d_vp;
    float *d_ln1, *d_t2, *d_input_ln;
    float *grad_norm_sq;
};

struct Model {
    Config cfg;
    P emb;
    Layer* L;
    P gf_p;
    float *final_ln, *xnf, *rmsf;
    float *logits, *d_logits, *loss_d;
    int *tok_d, *tgt_d;
    float *rope_cos, *rope_sin;
    float *buf1, *buf2;
    Workspace ws;
    cublasHandle_t cublas;
};

void init_rope(Model& m) {
    int hd = m.cfg.dk / 2, ml = m.cfg.max_seq;
    std::vector<float> ct(ml * hd), st(ml * hd);
    for (int p = 0; p < ml; ++p)
        for (int i = 0; i < hd; ++i) {
            double th = (double)p / pow(10000.0, 2.0 * (double)i / (double)m.cfg.dk);
            ct[p * hd + i] = (float)cos(th);
            st[p * hd + i] = (float)sin(th);
        }
    m.rope_cos = ga(ml * hd); h2d(m.rope_cos, ct.data(), ml * hd);
    m.rope_sin = ga(ml * hd); h2d(m.rope_sin, st.data(), ml * hd);
}

void init_model(Model& m, Config c) {
    m.cfg = c;
    CB(cublasCreate(&m.cublas));
    CB(cublasSetMathMode(m.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    float sd = sqrtf(2.0f / (float)c.dim);
    m.emb.init(c.vocab * c.dim); m.emb.rand_fill(0.02f);
    m.gf_p.init(c.dim); m.gf_p.fill_ones();
    m.L = new Layer[c.layers];
    int S = c.max_seq, D = c.dim, H = c.heads, dk = c.dk, F = c.dff;
    for (int i = 0; i < c.layers; ++i) {
        Layer& l = m.L[i];
        l.g1.init(D); l.g1.fill_ones();
        l.Wq.init(D * D); l.Wq.rand_fill(sd);
        l.Wk.init(D * D); l.Wk.rand_fill(sd);
        l.Wv.init(D * D); l.Wv.rand_fill(sd);
        l.Wo.init(D * D); l.Wo.rand_fill(sd);
        l.g2.init(D); l.g2.fill_ones();
        l.Wg.init(D * F); l.Wg.rand_fill(sd);
        l.Wu.init(D * F); l.Wu.rand_fill(sd);
        l.Wd.init(F * D); l.Wd.rand_fill(sqrtf(2.0f / (float)F));
        l.ln1_out = ga(S * D); l.xn1 = ga(S * D); l.rms1 = ga(S);
        l.q_heads = ga(H * S * dk); l.k_heads = ga(H * S * dk); l.v_heads = ga(H * S * dk);
        l.q_rope = ga(H * S * dk); l.k_rope = ga(H * S * dk);
        l.scores = ga(H * S * S); l.probs = ga(H * S * S);
        l.attn_out = ga(H * S * dk); l.concat = ga(S * D);
        l.residual = ga(S * D);
        l.ln2_out = ga(S * D); l.xn2 = ga(S * D); l.rms2 = ga(S);
        l.gate_pre = ga(S * F); l.up_pre = ga(S * F);
        l.gate_act = ga(S * F); l.hidden = ga(S * F);
    }
    m.final_ln = ga(S * D); m.xnf = ga(S * D); m.rmsf = ga(S);
    m.logits = ga(S * c.vocab); m.d_logits = ga(S * c.vocab);
    m.loss_d = ga(1);
    m.tok_d = nullptr; CK(cudaMalloc(&m.tok_d, S * sizeof(int)));
    m.tgt_d = nullptr; CK(cudaMalloc(&m.tgt_d, S * sizeof(int)));
    m.buf1 = ga(S * D); m.buf2 = ga(S * D);

    Workspace& w = m.ws;
    w.fwd_qvp = ga(S * D); w.fwd_kp = ga(S * D);
    w.fwd_proj = ga(S * D); w.fwd_ffn_out = ga(S * D);
    w.tmp_dh = ga(S * F);
    w.d_gate = ga(S * F); w.d_up = ga(S * F);
    w.d_ln2 = ga(S * D); w.d_ln2_2 = ga(S * D);
    w.d_res1_ln2 = ga(S * D);
    w.d_concat = ga(S * D);
    w.d_attn_out = ga(H * S * dk);
    w.d_probs = ga(H * S * S); w.d_scores = ga(H * S * S);
    w.d_V = ga(H * S * dk);
    w.d_Q_rope = ga(H * S * dk); w.d_K_rope = ga(H * S * dk);
    w.d_Q_heads = ga(H * S * dk); w.d_K_heads = ga(H * S * dk);
    w.d_qp = ga(S * D); w.d_kp = ga(S * D); w.d_vp = ga(S * D);
    w.d_ln1 = ga(S * D); w.d_t2 = ga(S * D); w.d_input_ln = ga(S * D);
    w.grad_norm_sq = ga(1);

    init_rope(m);
}

void forward(Model& m, int* tok_h, int seq, float* x_out) {
    int D = m.cfg.dim, H = m.cfg.heads, dk = m.cfg.dk;
    CK(cudaMemcpy(m.tok_d, tok_h, seq * sizeof(int), cudaMemcpyHostToDevice));
    int blk = 256;

    k_embed<<<DV(seq * D, blk), blk>>>(x_out, m.emb.w, m.tok_d, seq, D);

    Workspace& w = m.ws;
    for (int li = 0; li < m.cfg.layers; ++li) {
        Layer& l = m.L[li];
        k_rmsnorm<<<seq, blk, blk * sizeof(float)>>>(l.ln1_out, l.rms1, l.xn1, x_out, l.g1.w, seq, D);

        gemm_nn(m.cublas, l.ln1_out, l.Wq.w, w.fwd_qvp, seq, D, D);
        k_reshape_h<<<DV(H * seq * dk, blk), blk>>>(l.q_heads, w.fwd_qvp, seq, H, dk);

        gemm_nn(m.cublas, l.ln1_out, l.Wk.w, w.fwd_kp, seq, D, D);
        k_reshape_h<<<DV(H * seq * dk, blk), blk>>>(l.k_heads, w.fwd_kp, seq, H, dk);

        gemm_nn(m.cublas, l.ln1_out, l.Wv.w, w.fwd_qvp, seq, D, D);
        k_reshape_h<<<DV(H * seq * dk, blk), blk>>>(l.v_heads, w.fwd_qvp, seq, H, dk);

        int rope_n = H * seq * (dk / 2);
        k_rope<<<DV(rope_n, blk), blk>>>(l.q_rope, l.q_heads, m.rope_cos, m.rope_sin, H, seq, dk);
        k_rope<<<DV(rope_n, blk), blk>>>(l.k_rope, l.k_heads, m.rope_cos, m.rope_sin, H, seq, dk);

        long long sh = (long long)seq * dk, ss = (long long)seq * seq;
        float scale = 1.0f / sqrtf((float)dk);
        bgemm_nt_s(m.cublas, l.q_rope, l.k_rope, l.scores, seq, dk, seq, H, sh, sh, ss, scale);

        dim3 sg(seq, H);
        k_causal_softmax<<<sg, blk, blk * sizeof(float)>>>(l.probs, l.scores, H, seq);

        bgemm_nn(m.cublas, l.probs, l.v_heads, l.attn_out, seq, seq, dk, H, ss, sh, sh);

        k_unreshape_h<<<DV(H * seq * dk, blk), blk>>>(l.concat, l.attn_out, seq, H, dk);

        gemm_nn(m.cublas, l.concat, l.Wo.w, w.fwd_proj, seq, D, D);
        k_add<<<DV(seq * D, blk), blk>>>(l.residual, x_out, w.fwd_proj, seq * D);

        k_rmsnorm<<<seq, blk, blk * sizeof(float)>>>(l.ln2_out, l.rms2, l.xn2, l.residual, l.g2.w, seq, D);

        gemm_nn(m.cublas, l.ln2_out, l.Wg.w, l.gate_pre, seq, D, m.cfg.dff);
        gemm_nn(m.cublas, l.ln2_out, l.Wu.w, l.up_pre, seq, D, m.cfg.dff);
        int fn = seq * m.cfg.dff;
        k_silu_mul<<<DV(fn, blk), blk>>>(l.hidden, l.gate_act, l.gate_pre, l.up_pre, fn);

        gemm_nn(m.cublas, l.hidden, l.Wd.w, w.fwd_ffn_out, seq, m.cfg.dff, D);
        k_add<<<DV(seq * D, blk), blk>>>(x_out, l.residual, w.fwd_ffn_out, seq * D);
    }

    k_rmsnorm<<<seq, blk, blk * sizeof(float)>>>(m.final_ln, m.rmsf, m.xnf, x_out, m.gf_p.w, seq, D);
    gemm_nt(m.cublas, m.final_ln, m.emb.w, m.logits, seq, D, m.cfg.vocab);
}

float compute_loss_bwd(Model& m, int* tok_h, int* tgt_h, int seq) {
    int D = m.cfg.dim, H = m.cfg.heads, dk = m.cfg.dk, V = m.cfg.vocab, F = m.cfg.dff;
    int blk = 256;

    float* x = m.buf1;
    forward(m, tok_h, seq, x);

    CK(cudaMemcpy(m.tgt_d, tgt_h, seq * sizeof(int), cudaMemcpyHostToDevice));
    gz(m.loss_d, 1);
    k_ce_softmax<<<seq, blk, blk * sizeof(float)>>>(m.d_logits, m.loss_d, m.logits, m.tgt_d, seq, V);

    float loss_h;
    d2h(&loss_h, m.loss_d, 1);

    float* d_fln = m.buf2;
    gemm_nn(m.cublas, m.d_logits, m.emb.w, d_fln, seq, V, D);
    gemm_tn_acc(m.cublas, m.d_logits, m.final_ln, m.emb.g, seq, V, D);

    float* dx = m.buf1;
    k_rmsnorm_bwd<<<seq, blk, blk * sizeof(float)>>>(dx, m.gf_p.g, d_fln, m.xnf, m.rmsf, m.gf_p.w, seq, D);

    Workspace& w = m.ws;
    for (int li = m.cfg.layers - 1; li >= 0; --li) {
        Layer& l = m.L[li];

        float* d_ffn = m.buf2;
        d2d(d_ffn, dx, seq * D);
        float* d_res1 = dx;

        gemm_tn_acc(m.cublas, l.hidden, d_ffn, l.Wd.g, seq, F, D);
        gemm_nt(m.cublas, d_ffn, l.Wd.w, w.tmp_dh, seq, D, F);

        k_silu_mul_bwd<<<DV(seq * F, blk), blk>>>(w.d_gate, w.d_up, w.tmp_dh, l.gate_pre, l.up_pre, l.gate_act, seq * F);

        gemm_tn_acc(m.cublas, l.ln2_out, w.d_gate, l.Wg.g, seq, D, F);
        gemm_tn_acc(m.cublas, l.ln2_out, w.d_up, l.Wu.g, seq, D, F);

        gemm_nn(m.cublas, w.d_gate, l.Wg.w, w.d_ln2, seq, F, D);
        gemm_nn(m.cublas, w.d_up, l.Wu.w, w.d_ln2_2, seq, F, D);
        k_add<<<DV(seq * D, blk), blk>>>(w.d_ln2, w.d_ln2, w.d_ln2_2, seq * D);

        k_rmsnorm_bwd<<<seq, blk, blk * sizeof(float)>>>(w.d_res1_ln2, l.g2.g, w.d_ln2, l.xn2, l.rms2, l.g2.w, seq, D);
        k_add<<<DV(seq * D, blk), blk>>>(d_res1, d_res1, w.d_res1_ln2, seq * D);

        gemm_tn_acc(m.cublas, l.concat, d_res1, l.Wo.g, seq, D, D);
        gemm_nt(m.cublas, d_res1, l.Wo.w, w.d_concat, seq, D, D);

        k_reshape_h<<<DV(H * seq * dk, blk), blk>>>(w.d_attn_out, w.d_concat, seq, H, dk);

        long long sh = (long long)seq * dk, ss = (long long)seq * seq;

        bgemm_nt(m.cublas, w.d_attn_out, l.v_heads, w.d_probs, seq, dk, seq, H, sh, sh, ss);
        bgemm_tn(m.cublas, l.probs, w.d_attn_out, w.d_V, seq, seq, dk, H, ss, sh, sh);

        dim3 sg(seq, H);
        k_softmax_bwd<<<sg, blk, blk * sizeof(float)>>>(w.d_scores, w.d_probs, l.probs, H, seq);

        float scale = 1.0f / sqrtf((float)dk);
        int ns = H * seq * seq;
        CB(cublasSscal(m.cublas, ns, &scale, w.d_scores, 1));

        bgemm_nn(m.cublas, w.d_scores, l.k_rope, w.d_Q_rope, seq, seq, dk, H, ss, sh, sh);
        bgemm_tn(m.cublas, w.d_scores, l.q_rope, w.d_K_rope, seq, seq, dk, H, ss, sh, sh);

        int rope_n = H * seq * (dk / 2);
        k_rope_bwd<<<DV(rope_n, blk), blk>>>(w.d_Q_heads, w.d_Q_rope, m.rope_cos, m.rope_sin, H, seq, dk);
        k_rope_bwd<<<DV(rope_n, blk), blk>>>(w.d_K_heads, w.d_K_rope, m.rope_cos, m.rope_sin, H, seq, dk);

        k_unreshape_h<<<DV(H * seq * dk, blk), blk>>>(w.d_qp, w.d_Q_heads, seq, H, dk);
        k_unreshape_h<<<DV(H * seq * dk, blk), blk>>>(w.d_kp, w.d_K_heads, seq, H, dk);
        k_unreshape_h<<<DV(H * seq * dk, blk), blk>>>(w.d_vp, w.d_V, seq, H, dk);

        gemm_tn_acc(m.cublas, l.ln1_out, w.d_qp, l.Wq.g, seq, D, D);
        gemm_tn_acc(m.cublas, l.ln1_out, w.d_kp, l.Wk.g, seq, D, D);
        gemm_tn_acc(m.cublas, l.ln1_out, w.d_vp, l.Wv.g, seq, D, D);

        gemm_nn(m.cublas, w.d_qp, l.Wq.w, w.d_ln1, seq, D, D);
        gemm_nn(m.cublas, w.d_kp, l.Wk.w, w.d_t2, seq, D, D);
        k_add<<<DV(seq * D, blk), blk>>>(w.d_ln1, w.d_ln1, w.d_t2, seq * D);
        gemm_nn(m.cublas, w.d_vp, l.Wv.w, w.d_t2, seq, D, D);
        k_add<<<DV(seq * D, blk), blk>>>(w.d_ln1, w.d_ln1, w.d_t2, seq * D);

        k_rmsnorm_bwd<<<seq, blk, blk * sizeof(float)>>>(w.d_input_ln, l.g1.g, w.d_ln1, l.xn1, l.rms1, l.g1.w, seq, D);
        k_add<<<DV(seq * D, blk), blk>>>(dx, d_res1, w.d_input_ln, seq * D);
    }

    k_embed_bwd<<<DV(seq * D, blk), blk>>>(m.emb.g, dx, m.tok_d, seq, D);
    return loss_h;
}

float clip_gradients(Model& m, float max_norm) {
    gz(m.ws.grad_norm_sq, 1);
    int blk = 256;
    auto accum = [&](float* g, int n) {
        k_sqsum<<<DV(n, blk), blk, blk * sizeof(float)>>>(m.ws.grad_norm_sq, g, n);
    };
    accum(m.emb.g, m.emb.sz);
    accum(m.gf_p.g, m.gf_p.sz);
    for (int i = 0; i < m.cfg.layers; ++i) {
        Layer& l = m.L[i];
        accum(l.g1.g, l.g1.sz); accum(l.Wq.g, l.Wq.sz);
        accum(l.Wk.g, l.Wk.sz); accum(l.Wv.g, l.Wv.sz);
        accum(l.Wo.g, l.Wo.sz); accum(l.g2.g, l.g2.sz);
        accum(l.Wg.g, l.Wg.sz); accum(l.Wu.g, l.Wu.sz);
        accum(l.Wd.g, l.Wd.sz);
    }
    float sq_h;
    d2h(&sq_h, m.ws.grad_norm_sq, 1);
    float norm = sqrtf(sq_h);
    if (norm > max_norm && norm > 0.0f) {
        float scale = max_norm / (norm + 1e-6f);
        auto sc = [&](float* g, int n) {
            CB(cublasSscal(m.cublas, n, &scale, g, 1));
        };
        sc(m.emb.g, m.emb.sz);
        sc(m.gf_p.g, m.gf_p.sz);
        for (int i = 0; i < m.cfg.layers; ++i) {
            Layer& l = m.L[i];
            sc(l.g1.g, l.g1.sz); sc(l.Wq.g, l.Wq.sz);
            sc(l.Wk.g, l.Wk.sz); sc(l.Wv.g, l.Wv.sz);
            sc(l.Wo.g, l.Wo.sz); sc(l.g2.g, l.g2.sz);
            sc(l.Wg.g, l.Wg.sz); sc(l.Wu.g, l.Wu.sz);
            sc(l.Wd.g, l.Wd.sz);
        }
    }
    return norm;
}

void update_model(Model& m, float lr, float wd, int step) {
    float b1 = 0.9f, b2 = 0.98f, eps = 1e-9f;
    float bc1 = powf(b1, (float)step), bc2 = powf(b2, (float)step);
    m.emb.update(lr, b1, b2, eps, wd, bc1, bc2);
    m.gf_p.update(lr, b1, b2, eps, 0.0f, bc1, bc2);
    for (int i = 0; i < m.cfg.layers; ++i) {
        Layer& l = m.L[i];
        l.g1.update(lr, b1, b2, eps, 0.0f, bc1, bc2);
        l.Wq.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.Wk.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.Wv.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.Wo.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.g2.update(lr, b1, b2, eps, 0.0f, bc1, bc2);
        l.Wg.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.Wu.update(lr, b1, b2, eps, wd, bc1, bc2);
        l.Wd.update(lr, b1, b2, eps, wd, bc1, bc2);
    }
}

void zero_grad(Model& m) {
    m.emb.zero_grad(); m.gf_p.zero_grad();
    for (int i = 0; i < m.cfg.layers; ++i) {
        Layer& l = m.L[i];
        l.g1.zero_grad(); l.Wq.zero_grad(); l.Wk.zero_grad();
        l.Wv.zero_grad(); l.Wo.zero_grad(); l.g2.zero_grad();
        l.Wg.zero_grad(); l.Wu.zero_grad(); l.Wd.zero_grad();
    }
}

bool save_ckpt(Model& m, const char* path, int step, float loss) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite("XGPU", 1, 4, f);
    int v = 1; fwrite(&v, 4, 1, f);
    fwrite(&m.cfg, sizeof(Config), 1, f);
    fwrite(&step, 4, 1, f); fwrite(&loss, 4, 1, f);
    m.emb.save(f); m.gf_p.save(f);
    for (int i = 0; i < m.cfg.layers; ++i) {
        Layer& l = m.L[i];
        l.g1.save(f); l.Wq.save(f); l.Wk.save(f); l.Wv.save(f); l.Wo.save(f);
        l.g2.save(f); l.Wg.save(f); l.Wu.save(f); l.Wd.save(f);
    }
    fclose(f); return true;
}

bool load_ckpt(Model& m, const char* path, int& step, float& loss) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "XGPU", 4) != 0) { fclose(f); return false; }
    int v; fread(&v, 4, 1, f);
    Config c; fread(&c, sizeof(Config), 1, f);
    fread(&step, 4, 1, f); fread(&loss, 4, 1, f);
    init_model(m, c);
    m.emb.load(f); m.gf_p.load(f);
    for (int i = 0; i < c.layers; ++i) {
        Layer& l = m.L[i];
        l.g1.load(f); l.Wq.load(f); l.Wk.load(f); l.Wv.load(f); l.Wo.load(f);
        l.g2.load(f); l.Wg.load(f); l.Wu.load(f); l.Wd.load(f);
    }
    fclose(f); return true;
}

size_t total_params(Model& m) {
    size_t t = m.emb.sz + m.gf_p.sz;
    for (int i = 0; i < m.cfg.layers; ++i) {
        Layer& l = m.L[i];
        t += l.g1.sz + l.Wq.sz + l.Wk.sz + l.Wv.sz + l.Wo.sz;
        t += l.g2.sz + l.Wg.sz + l.Wu.sz + l.Wd.sz;
    }
    return t;
}

int sample_token(std::vector<float>& logits, float temp, int top_k, float top_p, std::mt19937& rng) {
    int V = (int)logits.size();
    if (temp <= 0.0f) {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    for (auto& v : logits) v /= temp;
    float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(V);
    float sm = 0.0f;
    for (int i = 0; i < V; ++i) { p[i] = expf(logits[i] - mx); sm += p[i]; }
    for (auto& v : p) v /= sm;

    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return p[a] > p[b]; });

    int keep = V;
    if (top_k > 0 && top_k < V) keep = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f; int cut = keep;
        for (int i = 0; i < keep; ++i) {
            cum += p[idx[i]];
            if (cum >= top_p) { cut = i + 1; break; }
        }
        keep = cut;
    }

    std::vector<float> w(keep);
    float s2 = 0.0f;
    for (int i = 0; i < keep; ++i) { w[i] = p[idx[i]]; s2 += w[i]; }
    for (auto& v : w) v /= s2;

    std::discrete_distribution<int> dist(w.begin(), w.end());
    return idx[dist(rng)];
}

std::vector<int> generate(Model& m, const std::vector<int>& prompt, int n_tokens,
                           float temp, int top_k, float top_p) {
    std::vector<int> result = prompt;
    std::mt19937 rng(std::random_device{}());
    int V = m.cfg.vocab;

    for (int t = 0; t < n_tokens; ++t) {
        int seq = (int)result.size();
        if (seq >= m.cfg.max_seq) break;
        forward(m, result.data(), seq, m.buf1);

        std::vector<float> log_h(V);
        d2h(log_h.data(), m.logits + (seq - 1) * V, V);

        result.push_back(sample_token(log_h, temp, top_k, top_p, rng));
    }
    return result;
}

enum class Mode { Train, Resume, Infer };

struct WizardCfg {
    Mode mode = Mode::Train;
    std::string data_path;
    std::string resume_path;
    int dim = 128, heads = 4, layers = 4, dff = 512, seq = 64;
    int total_steps = 10000;
    float lr = 3e-4f;
    bool valid = false;
};

static std::string ask(const std::string& prompt, const std::string& def) {
    std::cout << prompt;
    if (!def.empty()) std::cout << " [" << def << "]";
    std::cout << ": ";
    std::string line;
    if (!std::getline(std::cin, line)) return def;
    return line.empty() ? def : line;
}

static int ask_int(const std::string& prompt, int def) {
    std::string s = ask(prompt, std::to_string(def));
    try { return std::stoi(s); } catch (...) { return def; }
}

static float ask_float(const std::string& prompt, float def) {
    char buf[32]; snprintf(buf, sizeof(buf), "%g", def);
    std::string s = ask(prompt, buf);
    try { return std::stof(s); } catch (...) { return def; }
}

static std::vector<fs::path> list_corpus() {
    std::vector<fs::path> out;
    for (auto& e : fs::directory_iterator(".")) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".txt" || ext == ".md") out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::vector<fs::path> list_ckpts() {
    std::vector<fs::path> out;
    for (auto& e : fs::directory_iterator(".")) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension().string() == ".bin") out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static WizardCfg run_wizard() {
    WizardCfg c;
    std::cout << "\n";
    std::cout << "  =========================================\n";
    std::cout << "         XasmAI GPT - Interactive Mode\n";
    std::cout << "  =========================================\n\n";

    std::cout << "  Ne yapmak istersin?\n";
    std::cout << "    [1] Yeni model egit\n";
    std::cout << "    [2] Checkpoint'ten devam et\n";
    std::cout << "    [3] Mevcut modelle metin uret\n\n";
    int mc = ask_int("  Secim", 1);
    if (mc == 2) c.mode = Mode::Resume;
    else if (mc == 3) c.mode = Mode::Infer;
    else c.mode = Mode::Train;

    if (c.mode == Mode::Resume || c.mode == Mode::Infer) {
        auto ckpts = list_ckpts();
        if (ckpts.empty()) {
            std::cout << "\n  [!] Dizinde .bin checkpoint bulunamadi.\n";
            return c;
        }
        std::cout << "\n  Checkpoint dosyalari:\n";
        for (size_t i = 0; i < ckpts.size(); ++i) {
            auto sz = fs::file_size(ckpts[i]);
            std::cout << "    [" << (i + 1) << "] " << ckpts[i].filename().string()
                      << "  (" << (sz / 1024) << " KB)\n";
        }
        int pick = ask_int("  Secim", 1);
        if (pick < 1 || pick > (int)ckpts.size()) pick = 1;
        c.resume_path = ckpts[pick - 1].string();
    }

    auto corp = list_corpus();
    if (corp.empty()) {
        std::cout << "\n  [!] Dizinde .txt/.md corpus bulunamadi.\n";
        c.data_path = ask("\n  Corpus dosya yolu", "data.txt");
    } else {
        std::cout << "\n  Corpus dosyalari:\n";
        for (size_t i = 0; i < corp.size(); ++i) {
            auto sz = fs::file_size(corp[i]);
            std::cout << "    [" << (i + 1) << "] " << corp[i].filename().string()
                      << "  (" << (sz / 1024) << " KB)\n";
        }
        std::cout << "    [0] Baska bir yol gir\n";
        int pick = ask_int("  Secim", 1);
        if (pick == 0) c.data_path = ask("  Corpus yolu", "data.txt");
        else {
            if (pick < 1 || pick > (int)corp.size()) pick = 1;
            c.data_path = corp[pick - 1].string();
        }
    }

    if (c.mode == Mode::Train) {
        std::cout << "\n  Model boyutu (5070 Ti / 16GB VRAM):\n";
        std::cout << "    [1] Tiny    dim=128  heads=4  layers=4   (~1M params)\n";
        std::cout << "    [2] Small   dim=256  heads=8  layers=6   (~10M params)\n";
        std::cout << "    [3] Medium  dim=512  heads=8  layers=8   (~50M params)\n";
        std::cout << "    [4] Large   dim=768  heads=12 layers=12  (~150M params)\n";
        std::cout << "    [5] XL      dim=1024 heads=16 layers=16  (~400M params)\n";
        std::cout << "    [6] Ozel\n";
        int ps = ask_int("  Secim", 2);
        switch (ps) {
            case 1: c.dim=128;  c.heads=4;  c.layers=4;  c.dff=512;  c.seq=128; break;
            case 3: c.dim=512;  c.heads=8;  c.layers=8;  c.dff=2048; c.seq=256; break;
            case 4: c.dim=768;  c.heads=12; c.layers=12; c.dff=3072; c.seq=512; break;
            case 5: c.dim=1024; c.heads=16; c.layers=16; c.dff=4096; c.seq=512; break;
            case 6:
                c.dim    = ask_int("  dim", 256);
                c.heads  = ask_int("  heads", 8);
                c.layers = ask_int("  layers", 6);
                c.dff    = ask_int("  dff", c.dim * 4);
                c.seq    = ask_int("  seq (context)", 256);
                break;
            default: c.dim=256; c.heads=8; c.layers=6; c.dff=1024; c.seq=256; break;
        }
        c.total_steps = ask_int("\n  Toplam egitim adimi", 10000);
        c.lr          = ask_float("  Learning rate", 3e-4f);
    }

    c.valid = true;
    return c;
}

void run_inference_loop(Model& m, std::vector<char>& i2c,
                         std::map<char,int>& c2i, int top_k, float top_p) {
    std::cout << "\n  Uretim modu. Prompt yaz, Enter bas. '/q' = cik, '/temp 0.8' = temp ayarla, '/len 200' = uzunluk\n\n";
    float temp = 0.8f;
    int gen_len = 200;
    while (!g_stop) {
        std::cout << "  > ";
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line == "/q" || line == "/quit") break;
        if (line.rfind("/temp ", 0) == 0) {
            try { temp = std::stof(line.substr(6)); std::cout << "  temp = " << temp << "\n"; }
            catch (...) {} continue;
        }
        if (line.rfind("/len ", 0) == 0) {
            try { gen_len = std::stoi(line.substr(5)); std::cout << "  len = " << gen_len << "\n"; }
            catch (...) {} continue;
        }
        if (line.empty()) continue;
        std::vector<int> prompt;
        for (char ch : line) {
            auto it = c2i.find(ch);
            if (it != c2i.end()) prompt.push_back(it->second);
        }
        if (prompt.empty()) { std::cout << "  [vocab'da hic karakter yok]\n"; continue; }
        auto gen = generate(m, prompt, gen_len, temp, top_k, top_p);
        std::cout << "  ";
        for (int id : gen) if (id >= 0 && id < (int)i2c.size()) std::cout << i2c[id];
        std::cout << "\n\n";
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    std::string data_path = "data.txt", resume_path = "";
    int SEQ = 64, DIM = 128, HEADS = 4, LAYERS = 4, DFF = 512;
    float LR = 3e-4f, MIN_LR = 1e-5f, WD = 0.01f;
    int WARMUP = 1000, TOTAL = 10000000;
    int LOG_EVERY = 10, GEN_EVERY = 500, SAVE_EVERY = 5000, GEN_LEN = 80;
    float GRAD_CLIP = 1.0f;
    int TOP_K = 40;
    float TOP_P = 0.95f;
    std::string CKPT = "gpu_checkpoint";
    bool interactive = (argc == 1);
    bool infer_only = false;

    if (interactive) {
        WizardCfg wc = run_wizard();
        if (!wc.valid) {
            std::cout << "\n  [Iptal] Enter'a bas...\n";
            std::cin.get();
            return 1;
        }
        data_path = wc.data_path;
        if (wc.mode == Mode::Resume || wc.mode == Mode::Infer) resume_path = wc.resume_path;
        if (wc.mode == Mode::Infer) infer_only = true;
        if (wc.mode == Mode::Train) {
            SEQ = wc.seq; DIM = wc.dim; HEADS = wc.heads;
            LAYERS = wc.layers; DFF = wc.dff;
            LR = wc.lr; TOTAL = wc.total_steps;
            WARMUP = std::min(1000, TOTAL / 10);
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (a == "--seq" && i + 1 < argc) SEQ = atoi(argv[++i]);
        else if (a == "--dim" && i + 1 < argc) DIM = atoi(argv[++i]);
        else if (a == "--heads" && i + 1 < argc) HEADS = atoi(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) LAYERS = atoi(argv[++i]);
        else if (a == "--ff" && i + 1 < argc) DFF = atoi(argv[++i]);
        else if (a == "--lr" && i + 1 < argc) LR = atof(argv[++i]);
        else if (a == "--min-lr" && i + 1 < argc) MIN_LR = atof(argv[++i]);
        else if (a == "--wd" && i + 1 < argc) WD = atof(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) WARMUP = atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc) TOTAL = atoi(argv[++i]);
        else if (a == "--log-every" && i + 1 < argc) LOG_EVERY = atoi(argv[++i]);
        else if (a == "--gen-every" && i + 1 < argc) GEN_EVERY = atoi(argv[++i]);
        else if (a == "--save-every" && i + 1 < argc) SAVE_EVERY = atoi(argv[++i]);
        else if (a == "--ckpt" && i + 1 < argc) CKPT = argv[++i];
        else if (a == "--clip" && i + 1 < argc) GRAD_CLIP = atof(argv[++i]);
        else if (a == "--top-k" && i + 1 < argc) TOP_K = atoi(argv[++i]);
        else if (a == "--top-p" && i + 1 < argc) TOP_P = atof(argv[++i]);
        else if (a[0] != '-') data_path = a;
    }

    std::ifstream file(data_path);
    if (!file) { fprintf(stderr, "Dosya acilamadi: %s\n", data_path.c_str()); return 1; }
    std::string corpus((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::set<char> chars(corpus.begin(), corpus.end());
    std::map<char, int> c2i;
    std::vector<char> i2c;
    for (char c : chars) { c2i[c] = (int)i2c.size(); i2c.push_back(c); }
    int VOCAB = (int)i2c.size();
    std::vector<int> all_ids(corpus.size());
    for (size_t i = 0; i < corpus.size(); ++i) all_ids[i] = c2i[corpus[i]];

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("\n  GPU: %s | VRAM: %zu MB | SM: %d.%d\n\n",
           prop.name, prop.totalGlobalMem / (1024 * 1024), prop.major, prop.minor);

    Model m;
    int start_step = 0;
    float loss_ema = 0.0f;

    if (!resume_path.empty()) {
        if (load_ckpt(m, resume_path.c_str(), start_step, loss_ema)) {
            printf("  Checkpoint yuklendi: step=%d loss=%.4f\n", start_step, loss_ema);
        } else {
            fprintf(stderr, "Checkpoint yuklenemedi!\n"); return 1;
        }
    } else {
        Config cfg = { VOCAB, DIM, HEADS, LAYERS, DFF, SEQ, DIM / HEADS };
        init_model(m, cfg);
    }

    size_t tp = total_params(m);
    printf("  Model: dim=%d heads=%d layers=%d dff=%d seq=%d vocab=%d\n",
           m.cfg.dim, m.cfg.heads, m.cfg.layers, m.cfg.dff, m.cfg.max_seq, m.cfg.vocab);
    printf("  Params: %zu | Veri: %zu token\n\n", tp, all_ids.size());
    printf("  %8s %10s %8s %10s %10s %8s\n", "Step", "Loss", "PPL", "tok/s", "LR", "|g|");
    printf("  %s\n", std::string(68, '-').c_str());

    if (infer_only) {
        run_inference_loop(m, i2c, c2i, TOP_K, TOP_P);
        if (interactive) {
            std::cout << "\n  [Cikmak icin Enter]\n";
            std::cin.get();
        }
        return 0;
    }

    std::mt19937 rng(42);
    auto t0 = std::chrono::steady_clock::now();
    int tokens_since = 0;
    std::string ckpt_latest = CKPT + "_latest.bin";

    for (int step = start_step + 1; step <= TOTAL && !g_stop; ++step) {
        int max_start = (int)all_ids.size() - SEQ - 1;
        if (max_start < 0) max_start = 0;
        int pos = rng() % (max_start + 1);

        std::vector<int> input(all_ids.begin() + pos, all_ids.begin() + pos + SEQ);
        std::vector<int> target(all_ids.begin() + pos + 1, all_ids.begin() + pos + SEQ + 1);

        zero_grad(m);
        float loss = compute_loss_bwd(m, input.data(), target.data(), SEQ);

        if (!std::isfinite(loss)) {
            fprintf(stderr, "  [warn] non-finite loss at step %d, skipping update\n", step);
            continue;
        }

        float gnorm = clip_gradients(m, GRAD_CLIP);

        float lr = LR;
        if (step < WARMUP) lr = LR * (float)step / (float)WARMUP;
        else {
            float decay = 0.5f * (1.0f + cosf(3.14159265f * (float)(step - WARMUP) / (float)(TOTAL - WARMUP)));
            lr = MIN_LR + (LR - MIN_LR) * decay;
        }
        update_model(m, lr, WD, step);

        loss_ema = (step == start_step + 1) ? loss : 0.99f * loss_ema + 0.01f * loss;
        tokens_since += SEQ;

        if (step % LOG_EVERY == 0) {
            auto t1 = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            double tps = tokens_since / dt;
            printf("  %8d %10.4f %8.1f %10.0f %10.2e %8.2f\n",
                   step, loss_ema, expf(loss_ema), tps, lr, gnorm);
            tokens_since = 0; t0 = t1;
        }

        if (step % GEN_EVERY == 0) {
            int prompt_len = std::min(8, (int)all_ids.size());
            std::vector<int> prompt(all_ids.begin(), all_ids.begin() + prompt_len);
            auto gen = generate(m, prompt, GEN_LEN, 0.8f, TOP_K, TOP_P);
            std::string txt;
            for (int id : gen) if (id >= 0 && id < VOCAB) txt += i2c[id];
            printf("\n  >>> %s\n\n", txt.c_str());
        }

        if (step % SAVE_EVERY == 0) {
            save_ckpt(m, ckpt_latest.c_str(), step, loss_ema);
            printf("  [Checkpoint: %s]\n", ckpt_latest.c_str());
        }
    }

    if (g_stop) {
        save_ckpt(m, ckpt_latest.c_str(), start_step, loss_ema);
        printf("\n  [Emergency checkpoint saved]\n");
    }
    printf("\n  Egitim tamamlandi.\n");

    if (interactive) {
        std::cout << "\n  Simdi metin uretmek ister misin? [e/H]: ";
        std::string yn; std::getline(std::cin, yn);
        if (!yn.empty() && (yn[0] == 'e' || yn[0] == 'E' || yn[0] == 'y' || yn[0] == 'Y')) {
            run_inference_loop(m, i2c, c2i, TOP_K, TOP_P);
        }
        std::cout << "\n  [Cikmak icin Enter]\n";
        std::cin.get();
    }
    return 0;
}
