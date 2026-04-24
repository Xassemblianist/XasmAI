#pragma once
/*
 * XasmAI — Activation Functions
 * Sigmoid, ReLU, LeakyReLU, Tanh, Softmax
 * Her birinin türevi dahil (backpropagation için).
 */

#include "matrix.hpp"
#include <cmath>
#include <string>

namespace xasm {

enum class Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
    Linear    // aktivasyon yok (çıkış katmanı regresyon için)
};

namespace act {

    // ====== Sigmoid ======
    // f(x) = 1 / (1 + e^(-x))
    // Çıkış: (0, 1) — binary sınıflandırma için ideal
    inline Matrix sigmoid(const Matrix& x) {
        return x.apply([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
    }

    inline Matrix sigmoid_derivative(const Matrix& output) {
        // σ'(x) = σ(x) * (1 - σ(x))  — output zaten sigmoid çıktısı
        return output.apply([](double v) {
            return v * (1.0 - v);
        });
    }

    // ====== ReLU ======
    // f(x) = max(0, x)
    // Hızlı ve etkili — gizli katmanlar için varsayılan seçim
    inline Matrix relu(const Matrix& x) {
        return x.apply([](double v) {
            return v > 0.0 ? v : 0.0;
        });
    }

    inline Matrix relu_derivative(const Matrix& output) {
        return output.apply([](double v) {
            return v > 0.0 ? 1.0 : 0.0;
        });
    }

    // ====== Leaky ReLU ======
    // f(x) = max(0.01x, x)
    // "Dying ReLU" problemini çözer
    inline Matrix leaky_relu(const Matrix& x, double alpha = 0.01) {
        return x.apply([alpha](double v) {
            return v > 0.0 ? v : alpha * v;
        });
    }

    inline Matrix leaky_relu_derivative(const Matrix& output, double alpha = 0.01) {
        return output.apply([alpha](double v) {
            return v > 0.0 ? 1.0 : alpha;
        });
    }

    // ====== Tanh ======
    // f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    // Çıkış: (-1, 1)
    inline Matrix tanh_act(const Matrix& x) {
        return x.apply([](double v) {
            return std::tanh(v);
        });
    }

    inline Matrix tanh_derivative(const Matrix& output) {
        // tanh'(x) = 1 - tanh²(x)
        return output.apply([](double v) {
            return 1.0 - v * v;
        });
    }

    // ====== Softmax ======
    // f(xi) = e^xi / Σ(e^xj)
    // Çoklu sınıflandırma çıkış katmanı (olasılık dağılımı)
    inline Matrix softmax(const Matrix& x) {
        Matrix result(x.rows, x.cols);
        for (size_t i = 0; i < x.rows; ++i) {
            double max_v = x(i, 0);
            for (size_t j = 1; j < x.cols; ++j) {
                if (x(i, j) > max_v) max_v = x(i, j);
            }
            double sum = 0.0;
            for (size_t j = 0; j < x.cols; ++j) {
                result(i, j) = std::exp(x(i, j) - max_v);
                sum += result(i, j);
            }
            for (size_t j = 0; j < x.cols; ++j) {
                result(i, j) /= sum;
            }
        }
        return result;
    }

    // Softmax türevi (Jacobian basitleştirilmiş — cross-entropy ile kullanılır)
    inline Matrix softmax_derivative(const Matrix& output) {
        // CrossEntropy ile birlikte kullanıldığında: dL/dz = output - target
        // Bu yüzden burada sadece 1 döndürüyoruz (loss fonksiyonunda hesaplanır)
        return Matrix::ones(output.rows, output.cols);
    }

    // ====== Linear ======
    inline Matrix linear(const Matrix& x) { return x; }
    inline Matrix linear_derivative(const Matrix& output) {
        return Matrix::ones(output.rows, output.cols);
    }

    // ====== Dispatch ======
    // Aktivasyon tipine göre ileri yönlü hesaplama
    inline Matrix forward(Activation type, const Matrix& x) {
        switch (type) {
            case Activation::Sigmoid:   return sigmoid(x);
            case Activation::ReLU:      return relu(x);
            case Activation::LeakyReLU: return leaky_relu(x);
            case Activation::Tanh:      return tanh_act(x);
            case Activation::Softmax:   return softmax(x);
            case Activation::Linear:    return linear(x);
        }
        return x;
    }

    // Aktivasyon tipine göre türev
    inline Matrix derivative(Activation type, const Matrix& output) {
        switch (type) {
            case Activation::Sigmoid:   return sigmoid_derivative(output);
            case Activation::ReLU:      return relu_derivative(output);
            case Activation::LeakyReLU: return leaky_relu_derivative(output);
            case Activation::Tanh:      return tanh_derivative(output);
            case Activation::Softmax:   return softmax_derivative(output);
            case Activation::Linear:    return linear_derivative(output);
        }
        return output;
    }

    // Aktivasyon adı
    inline std::string name(Activation type) {
        switch (type) {
            case Activation::Sigmoid:   return "Sigmoid";
            case Activation::ReLU:      return "ReLU";
            case Activation::LeakyReLU: return "LeakyReLU";
            case Activation::Tanh:      return "Tanh";
            case Activation::Softmax:   return "Softmax";
            case Activation::Linear:    return "Linear";
        }
        return "Unknown";
    }

} // namespace act
} // namespace xasm
