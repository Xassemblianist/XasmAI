#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void xasm_cuda_init(int device_id);
void xasm_cuda_sync();

void* xasm_cuda_malloc(size_t bytes);
void xasm_cuda_free(void* ptr);
void xasm_cuda_memset(void* ptr, int value, size_t bytes);

void xasm_cuda_upload_fp16(void* d_dst, const double* h_src, size_t n);
void xasm_cuda_download_fp16(double* h_dst, const void* d_src, size_t n);

void xasm_cuda_gemm(const void* A, const void* B, void* C,
                     int M, int N, int K);

void xasm_cuda_gemm_add(const void* A, const void* B, void* C,
                          int M, int N, int K, const void* bias, int bias_cols);

void xasm_cuda_softmax(void* data, int rows, int cols);

void xasm_cuda_layernorm(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int rows, int cols, float eps);

void xasm_cuda_rope(void* Q, void* K, int seq_len, int dim,
                     int start_pos);

void xasm_cuda_add(const void* A, const void* B, void* C, size_t n);
void xasm_cuda_scale(void* data, size_t n, float scale);
void xasm_cuda_hadamard(const void* A, const void* B, void* C, size_t n);
void xasm_cuda_transpose(const void* src, void* dst, int rows, int cols);

int xasm_cuda_has_inf_nan(const void* data, size_t n);

void xasm_cuda_causal_mask(void* scores, int rows, int cols, int start_pos);

size_t xasm_cuda_mem_used();
size_t xasm_cuda_mem_total();

void xasm_cuda_device_info();

#ifdef __cplusplus
}
#endif
