#pragma once
/*
 * XasmAI — Matrix Library
 * Temel matris işlemleri: toplama, çıkarma, çarpma, transpose, vb.
 * Tüm sinir ağı hesaplamalarının temeli.
 */

#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <fstream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace xasm {

class Matrix {
public:
    size_t rows, cols;
    std::vector<double> data;

    // === Constructors ===

    Matrix() : rows(0), cols(0) {}

    Matrix(size_t r, size_t c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    Matrix(size_t r, size_t c, std::initializer_list<double> init)
        : rows(r), cols(c), data(init) {
        assert(data.size() == r * c);
    }

    // Vector'den matris (sütun vektörü olarak)
    static Matrix from_vector(const std::vector<double>& vec) {
        Matrix m(vec.size(), 1);
        m.data = vec;
        return m;
    }

    // Tek satır matris
    static Matrix row_vector(const std::vector<double>& vec) {
        Matrix m(1, vec.size());
        m.data = vec;
        return m;
    }

    // === Element Access ===

    double& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    double& at(size_t r, size_t c) {
        if (r >= rows || c >= cols)
            throw std::out_of_range("Matrix index out of range");
        return data[r * cols + c];
    }

    // === Rastgele Başlatma ===

    // Xavier/Glorot initialization — sigmoid/tanh için ideal
    static Matrix xavier(size_t r, size_t c) {
        static std::mt19937 gen(std::random_device{}());
        double limit = std::sqrt(6.0 / (r + c));
        std::uniform_real_distribution<double> dist(-limit, limit);
        Matrix m(r, c);
        for (auto& v : m.data) v = dist(gen);
        return m;
    }

    // He initialization — ReLU için ideal
    static Matrix he(size_t r, size_t c) {
        static std::mt19937 gen(std::random_device{}());
        double stddev = std::sqrt(2.0 / r);
        std::normal_distribution<double> dist(0.0, stddev);
        Matrix m(r, c);
        for (auto& v : m.data) v = dist(gen);
        return m;
    }

    // Rastgele uniform [-range, range]
    static Matrix random(size_t r, size_t c, double range = 1.0) {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-range, range);
        Matrix m(r, c);
        for (auto& v : m.data) v = dist(gen);
        return m;
    }

    // Sıfır matris
    static Matrix zeros(size_t r, size_t c) {
        return Matrix(r, c, 0.0);
    }

    // Bir matris
    static Matrix ones(size_t r, size_t c) {
        return Matrix(r, c, 1.0);
    }

    // Birim matris
    static Matrix identity(size_t n) {
        Matrix m(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            m(i, i) = 1.0;
        return m;
    }

    // === Aritmetik Operatörler ===

    // Matris + Matris (element-wise)
    Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // Matris - Matris (element-wise)
    Matrix operator-(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);
        Matrix result(rows, other.cols, 0.0);
#ifdef __AVX2__
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < cols; ++k) {
                __m256d a_vec = _mm256_set1_pd(data[i * cols + k]);
                const double* b_row = &other.data[k * other.cols];
                double* r_row = &result.data[i * other.cols];
                size_t j = 0;
                for (; j + 4 <= other.cols; j += 4) {
                    __m256d b = _mm256_loadu_pd(b_row + j);
                    __m256d c = _mm256_loadu_pd(r_row + j);
                    _mm256_storeu_pd(r_row + j, _mm256_fmadd_pd(a_vec, b, c));
                }
                double a_s = data[i * cols + k];
                for (; j < other.cols; ++j)
                    r_row[j] += a_s * b_row[j];
            }
#else
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < cols; ++k) {
                double a = data[i * cols + k];
                for (size_t j = 0; j < other.cols; ++j)
                    result.data[i * other.cols + j] += a * other.data[k * other.cols + j];
            }
#endif
        return result;
    }

    // Matris * Skalar
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    // Skalar * Matris
    friend Matrix operator*(double scalar, const Matrix& m) {
        return m * scalar;
    }

    // Matris / Skalar
    Matrix operator/(double scalar) const {
        return *this * (1.0 / scalar);
    }

    // += operatörü
    Matrix& operator+=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] += other.data[i];
        return *this;
    }

    // -= operatörü
    Matrix& operator-=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] -= other.data[i];
        return *this;
    }

    // Negatif
    Matrix operator-() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = -data[i];
        return result;
    }

    // === Matris İşlemleri ===

    // Transpose
    Matrix T() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }

    // Hadamard çarpımı (element-wise)
    Matrix hadamard(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] * other.data[i];
        return result;
    }

    // Her elemana fonksiyon uygula
    Matrix apply(std::function<double(double)> fn) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = fn(data[i]);
        return result;
    }

    // Topla — tüm elemanlar
    double sum() const {
        return std::accumulate(data.begin(), data.end(), 0.0);
    }

    // Ortalama
    double mean() const {
        return sum() / static_cast<double>(data.size());
    }

    // Maksimum değer
    double max_val() const {
        return *std::max_element(data.begin(), data.end());
    }

    // Minimum değer
    double min_val() const {
        return *std::min_element(data.begin(), data.end());
    }

    // Max index (argmax)
    size_t argmax() const {
        return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    }

    // Sütunlar boyunca topla → satır vektörü
    Matrix sum_rows() const {
        Matrix result(1, cols, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(0, j) += (*this)(i, j);
        return result;
    }

    // Satırlar boyunca topla → sütun vektörü
    Matrix sum_cols() const {
        Matrix result(rows, 1, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(i, 0) += (*this)(i, j);
        return result;
    }

    // Boyut
    size_t size() const { return data.size(); }

    // std::vector'e dönüştür
    std::vector<double> to_vector() const { return data; }

    // Flatten (1D vektöre)
    Matrix flatten() const {
        Matrix result(data.size(), 1);
        result.data = data;
        return result;
    }

    // Yeniden boyutlandır
    Matrix reshape(size_t r, size_t c) const {
        assert(r * c == data.size());
        Matrix result(r, c);
        result.data = data;
        return result;
    }

    // Clip (değerleri min/max arasında tut)
    Matrix clip(double min_val, double max_val) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::clamp(data[i], min_val, max_val);
        return result;
    }

    // === Yazdırma ===

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "Matrix(" << m.rows << "x" << m.cols << ") [\n";
        for (size_t i = 0; i < m.rows; ++i) {
            os << "  [";
            for (size_t j = 0; j < m.cols; ++j) {
                os << std::fixed << std::setprecision(4) << m(i, j);
                if (j < m.cols - 1) os << ", ";
            }
            os << "]";
            if (i < m.rows - 1) os << ",";
            os << "\n";
        }
        os << "]";
        return os;
    }

    void serialize(std::ostream& os) const {
        uint64_t r = rows, c = cols;
        os.write(reinterpret_cast<const char*>(&r), 8);
        os.write(reinterpret_cast<const char*>(&c), 8);
        os.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(double)));
    }

    static Matrix deserialize(std::istream& is) {
        uint64_t r, c;
        is.read(reinterpret_cast<char*>(&r), 8);
        is.read(reinterpret_cast<char*>(&c), 8);
        Matrix m(static_cast<size_t>(r), static_cast<size_t>(c));
        is.read(reinterpret_cast<char*>(m.data.data()),
                static_cast<std::streamsize>(r * c * sizeof(double)));
        return m;
    }

    // Kısa yazdırma
    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ": ";
        std::cout << *this << "\n";
    }

#ifdef XASM_CUDA
    void* d_ptr = nullptr;
    bool on_gpu = false;

    void to_gpu() {
        if (!d_ptr) d_ptr = xasm_cuda_malloc(data.size() * 2);
        xasm_cuda_upload_fp16(d_ptr, data.data(), data.size());
        on_gpu = true;
    }

    void from_gpu() {
        if (d_ptr && on_gpu)
            xasm_cuda_download_fp16(data.data(), d_ptr, data.size());
        on_gpu = false;
    }

    void free_gpu() {
        if (d_ptr) { xasm_cuda_free(d_ptr); d_ptr = nullptr; }
        on_gpu = false;
    }

    void ensure_gpu() {
        if (!on_gpu) to_gpu();
    }

    void ensure_cpu() {
        if (on_gpu) from_gpu();
    }

    void* gpu_alloc_result(size_t r, size_t c) {
        return xasm_cuda_malloc(r * c * 2);
    }

    Matrix gpu_matmul(const Matrix& other) const {
        Matrix result(rows, other.cols, 0.0);
        result.d_ptr = xasm_cuda_malloc(result.data.size() * 2);
        xasm_cuda_gemm(d_ptr, other.d_ptr, result.d_ptr,
                        static_cast<int>(rows),
                        static_cast<int>(other.cols),
                        static_cast<int>(cols));
        result.on_gpu = true;
        return result;
    }

    Matrix gpu_transpose() const {
        Matrix result(cols, rows, 0.0);
        result.d_ptr = xasm_cuda_malloc(result.data.size() * 2);
        xasm_cuda_transpose(d_ptr, result.d_ptr,
                              static_cast<int>(rows),
                              static_cast<int>(cols));
        result.on_gpu = true;
        return result;
    }

    Matrix gpu_add(const Matrix& other) const {
        Matrix result(rows, cols, 0.0);
        result.d_ptr = xasm_cuda_malloc(result.data.size() * 2);
        xasm_cuda_add(d_ptr, other.d_ptr, result.d_ptr, data.size());
        result.on_gpu = true;
        return result;
    }

    Matrix gpu_hadamard(const Matrix& other) const {
        Matrix result(rows, cols, 0.0);
        result.d_ptr = xasm_cuda_malloc(result.data.size() * 2);
        xasm_cuda_hadamard(d_ptr, other.d_ptr, result.d_ptr, data.size());
        result.on_gpu = true;
        return result;
    }

    void gpu_scale(float s) {
        xasm_cuda_scale(d_ptr, data.size(), s);
    }

    bool gpu_has_overflow() const {
        return xasm_cuda_has_inf_nan(d_ptr, data.size()) != 0;
    }

    ~Matrix() {
        if (d_ptr) xasm_cuda_free(d_ptr);
    }

    Matrix(const Matrix& other)
        : rows(other.rows), cols(other.cols), data(other.data),
          d_ptr(nullptr), on_gpu(false) {
        if (other.d_ptr && other.on_gpu) {
            d_ptr = xasm_cuda_malloc(data.size() * 2);
            cudaMemcpy(d_ptr, other.d_ptr, data.size() * 2, cudaMemcpyDeviceToDevice);
            on_gpu = true;
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (d_ptr) xasm_cuda_free(d_ptr);
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            d_ptr = nullptr;
            on_gpu = false;
            if (other.d_ptr && other.on_gpu) {
                d_ptr = xasm_cuda_malloc(data.size() * 2);
                cudaMemcpy(d_ptr, other.d_ptr, data.size() * 2, cudaMemcpyDeviceToDevice);
                on_gpu = true;
            }
        }
        return *this;
    }

    Matrix(Matrix&& other) noexcept
        : rows(other.rows), cols(other.cols), data(std::move(other.data)),
          d_ptr(other.d_ptr), on_gpu(other.on_gpu) {
        other.d_ptr = nullptr;
        other.on_gpu = false;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (d_ptr) xasm_cuda_free(d_ptr);
            rows = other.rows;
            cols = other.cols;
            data = std::move(other.data);
            d_ptr = other.d_ptr;
            on_gpu = other.on_gpu;
            other.d_ptr = nullptr;
            other.on_gpu = false;
        }
        return *this;
    }
#endif
};

} // namespace xasm

