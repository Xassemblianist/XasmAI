#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void kernel_double_to_half(const double* __restrict__ src,
                                       half* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(static_cast<float>(src[idx]));
}

__global__ void kernel_half_to_double(const half* __restrict__ src,
                                       double* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<double>(__half2float(src[idx]));
}

__global__ void gemm_wmma_kernel(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   half* __restrict__ C,
                                   int M, int N, int K) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int totalWarps = (gridDim.x * blockDim.x) / WARP_SIZE;

    int tilesM = (M + WMMA_M - 1) / WMMA_M;
    int tilesN = (N + WMMA_N - 1) / WMMA_N;

    for (int tile = warpId; tile < tilesM * tilesN; tile += totalWarps) {
        int tileRow = tile / tilesN;
        int tileCol = tile % tilesN;

        int rowStart = tileRow * WMMA_M;
        int colStart = tileCol * WMMA_N;

        if (rowStart >= M || colStart >= N) continue;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k = 0; k < K; k += WMMA_K) {
            if (rowStart + WMMA_M <= M && k + WMMA_K <= K) {
                wmma::load_matrix_sync(a_frag, A + rowStart * K + k, K);
            } else {
                wmma::fill_fragment(a_frag, __float2half(0.0f));
                for (int i = 0; i < WMMA_M; ++i)
                    for (int j = 0; j < WMMA_K; ++j)
                        if (rowStart + i < M && k + j < K)
                            a_frag.x[i * WMMA_K + j] = A[(rowStart + i) * K + k + j];
            }

            if (k + WMMA_K <= K && colStart + WMMA_N <= N) {
                wmma::load_matrix_sync(b_frag, B + k * N + colStart, N);
            } else {
                wmma::fill_fragment(b_frag, __float2half(0.0f));
                for (int i = 0; i < WMMA_K; ++i)
                    for (int j = 0; j < WMMA_N; ++j)
                        if (k + i < K && colStart + j < N)
                            b_frag.x[i * WMMA_N + j] = B[(k + i) * N + colStart + j];
            }

            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        if (rowStart + WMMA_M <= M && colStart + WMMA_N <= N) {
            half* out_ptr = C + rowStart * N + colStart;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out_frag;
            for (int i = 0; i < acc.num_elements; ++i)
                out_frag.x[i] = __float2half(acc.x[i]);
            wmma::store_matrix_sync(out_ptr, out_frag, N, wmma::mem_row_major);
        } else {
            float tmp[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(tmp, acc, WMMA_N, wmma::mem_row_major);
            for (int i = 0; i < WMMA_M; ++i)
                for (int j = 0; j < WMMA_N; ++j)
                    if (rowStart + i < M && colStart + j < N)
                        C[(rowStart + i) * N + colStart + j] = __float2half(tmp[i * WMMA_N + j]);
        }
    }
}

__global__ void gemm_naive_fp16(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  half* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    C[row * N + col] = __float2half(sum);
}

__global__ void softmax_kernel(half* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    half* row_ptr = data + row * cols;

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __half2float(row_ptr[j]);
        smem[j] = v;
        local_max = fmaxf(local_max, v);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));

    __shared__ float block_max;
    if (threadIdx.x % WARP_SIZE == 0)
        atomicMax((int*)&block_max, __float_as_int(local_max));
    if (threadIdx.x == 0) block_max = local_max;
    __syncthreads();

    float mx = block_max;
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = expf(smem[j] - mx);
        smem[j] = v;
        local_sum += v;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0)
        atomicAdd(&block_sum, local_sum);
    __syncthreads();

    float inv_sum = 1.0f / block_sum;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_ptr[j] = __float2half(smem[j] * inv_sum);
}

__global__ void layernorm_kernel(const half* __restrict__ input,
                                    half* __restrict__ output,
                                    const half* __restrict__ gamma,
                                    const half* __restrict__ beta,
                                    int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* in_row = input + row * cols;
    half* out_row = output + row * cols;

    float mean = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        mean += __half2float(in_row[j]);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);

    __shared__ float smean;
    if (threadIdx.x == 0) smean = 0.0f;
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) atomicAdd(&smean, mean);
    __syncthreads();
    mean = smean / static_cast<float>(cols);

    float var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = __half2float(in_row[j]) - mean;
        var += d * d;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        var += __shfl_xor_sync(0xFFFFFFFF, var, offset);

    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.0f;
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) atomicAdd(&svar, var);
    __syncthreads();
    var = svar / static_cast<float>(cols);

    float inv_std = rsqrtf(var + eps);

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float x_norm = (__half2float(in_row[j]) - mean) * inv_std;
        float y = x_norm * __half2float(gamma[j]) + __half2float(beta[j]);
        out_row[j] = __float2half(y);
    }
}

__global__ void rope_kernel(half* Q, half* K,
                              int seq_len, int dim, int start_pos) {
    int pos = blockIdx.x;
    int i = threadIdx.x;

    if (pos >= seq_len || 2 * i + 1 >= dim) return;

    int abs_pos = start_pos + pos;
    float theta = static_cast<float>(abs_pos) /
        powf(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(dim));
    float c = cosf(theta);
    float s = sinf(theta);

    int idx0 = pos * dim + 2 * i;
    int idx1 = pos * dim + 2 * i + 1;

    float q0 = __half2float(Q[idx0]);
    float q1 = __half2float(Q[idx1]);
    Q[idx0] = __float2half(q0 * c - q1 * s);
    Q[idx1] = __float2half(q0 * s + q1 * c);

    float k0 = __half2float(K[idx0]);
    float k1 = __half2float(K[idx1]);
    K[idx0] = __float2half(k0 * c - k1 * s);
    K[idx1] = __float2half(k0 * s + k1 * c);
}

__global__ void add_kernel(const half* A, const half* B, half* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = __float2half(__half2float(A[idx]) + __half2float(B[idx]));
}

__global__ void scale_kernel(half* data, size_t n, float s) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = __float2half(__half2float(data[idx]) * s);
}

__global__ void hadamard_kernel(const half* A, const half* B, half* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = __float2half(__half2float(A[idx]) * __half2float(B[idx]));
}

__global__ void transpose_kernel(const half* src, half* dst, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    dst[c * rows + r] = src[r * cols + c];
}

__global__ void causal_mask_kernel(half* scores, int rows, int cols, int start_pos) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    if (c > start_pos + r)
        scores[r * cols + c] = __float2half(-1e4f);
}

__global__ void check_inf_nan_kernel(const half* data, size_t n, int* flag) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(data[idx]);
    if (isinf(v) || isnan(v))
        atomicExch(flag, 1);
}

extern "C" {

void xasm_cuda_init(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "  GPU: %s | SM %d.%d | VRAM: %zu MB | SMs: %d\n",
            prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024 * 1024), prop.multiProcessorCount);
}

void xasm_cuda_sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

void* xasm_cuda_malloc(size_t bytes) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void xasm_cuda_free(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void xasm_cuda_memset(void* ptr, int value, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

void xasm_cuda_upload_fp16(void* d_dst, const double* h_src, size_t n) {
    double* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_tmp, h_src, n * sizeof(double), cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    kernel_double_to_half<<<blocks, threads>>>(d_tmp, (half*)d_dst, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_tmp));
}

void xasm_cuda_download_fp16(double* h_dst, const void* d_src, size_t n) {
    double* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(double)));
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    kernel_half_to_double<<<blocks, threads>>>((const half*)d_src, d_tmp, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst, d_tmp, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp));
}

void xasm_cuda_gemm(const void* A, const void* B, void* C,
                     int M, int N, int K) {
    if (M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0) {
        int tilesM = M / WMMA_M;
        int tilesN = N / WMMA_N;
        int totalTiles = tilesM * tilesN;
        int warpsPerBlock = 4;
        int threadsPerBlock = warpsPerBlock * WARP_SIZE;
        int numBlocks = (totalTiles + warpsPerBlock - 1) / warpsPerBlock;
        gemm_wmma_kernel<<<numBlocks, threadsPerBlock>>>
            ((const half*)A, (const half*)B, (half*)C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_naive_fp16<<<grid, block>>>
            ((const half*)A, (const half*)B, (half*)C, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_softmax(void* data, int rows, int cols) {
    int threads = min(256, cols);
    size_t smem = cols * sizeof(float);
    softmax_kernel<<<rows, threads, smem>>>((half*)data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_layernorm(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int rows, int cols, float eps) {
    int threads = min(256, cols);
    layernorm_kernel<<<rows, threads>>>
        ((const half*)input, (half*)output,
         (const half*)gamma, (const half*)beta,
         rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_rope(void* Q, void* K, int seq_len, int dim, int start_pos) {
    int half_dim = dim / 2;
    rope_kernel<<<seq_len, half_dim>>>
        ((half*)Q, (half*)K, seq_len, dim, start_pos);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_add(const void* A, const void* B, void* C, size_t n) {
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>((const half*)A, (const half*)B, (half*)C, n);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_scale(void* data, size_t n, float scale) {
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>((half*)data, n, scale);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_hadamard(const void* A, const void* B, void* C, size_t n) {
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    hadamard_kernel<<<blocks, threads>>>((const half*)A, (const half*)B, (half*)C, n);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_transpose(const void* src, void* dst, int rows, int cols) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_kernel<<<blocks, threads>>>((const half*)src, (half*)dst, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void xasm_cuda_causal_mask(void* scores, int rows, int cols, int start_pos) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    causal_mask_kernel<<<grid, block>>>((half*)scores, rows, cols, start_pos);
    CUDA_CHECK(cudaGetLastError());
}

int xasm_cuda_has_inf_nan(const void* data, size_t n) {
    int* d_flag;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;
    check_inf_nan_kernel<<<blocks, threads>>>((const half*)data, n, d_flag);
    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_flag));
    return h_flag;
}

size_t xasm_cuda_mem_used() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem - free_mem;
}

size_t xasm_cuda_mem_total() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem;
}

void xasm_cuda_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr,
        "  ┌─────────────────────────────────────────┐\n"
        "  │ %s\n"
        "  ├─────────────────────────────────────────┤\n"
        "  │ SM:          %d.%d                       \n"
        "  │ SMs:         %d                          \n"
        "  │ VRAM:        %zu MB                      \n"
        "  │ Clock:       %d MHz                      \n"
        "  │ Tensor:      %s                          \n"
        "  └─────────────────────────────────────────┘\n",
        prop.name, prop.major, prop.minor,
        prop.multiProcessorCount,
        prop.totalGlobalMem / (1024 * 1024),
        prop.clockRate / 1000,
        (prop.major >= 7) ? "YES" : "NO");
}

}
