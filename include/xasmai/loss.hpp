#pragma once
/*
 * XasmAI — Loss Functions
 * MSE, CrossEntropy, BinaryCrossEntropy
 * Her birinin türevi dahil.
 */

#include "matrix.hpp"
#include <cmath>
#include <string>

namespace xasm {

enum class Loss {
    MSE,                // Mean Squared Error — regresyon
    BinaryCrossEntropy, // Binary Cross-Entropy — ikili sınıflandırma
    CrossEntropy        // Categorical Cross-Entropy — çoklu sınıflandırma
};

namespace loss {

    constexpr double EPS = 1e-8;  // Numerical stability

    // ====== MSE (Mean Squared Error) ======
    // L = (1/n) * Σ(y_pred - y_true)²
    inline double mse(const Matrix& predicted, const Matrix& target) {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            double diff = predicted.data[i] - target.data[i];
            sum += diff * diff;
        }
        return sum / static_cast<double>(predicted.data.size());
    }

    inline Matrix mse_derivative(const Matrix& predicted, const Matrix& target) {
        // dL/dy_pred = 2 * (y_pred - y_true) / n
        double n = static_cast<double>(predicted.data.size());
        Matrix result(predicted.rows, predicted.cols);
        for (size_t i = 0; i < predicted.data.size(); ++i)
            result.data[i] = 2.0 * (predicted.data[i] - target.data[i]) / n;
        return result;
    }

    // ====== Binary Cross-Entropy ======
    // L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    inline double binary_cross_entropy(const Matrix& predicted, const Matrix& target) {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            double p = std::clamp(predicted.data[i], EPS, 1.0 - EPS);
            sum += target.data[i] * std::log(p) + (1.0 - target.data[i]) * std::log(1.0 - p);
        }
        return -sum / static_cast<double>(predicted.data.size());
    }

    inline Matrix binary_cross_entropy_derivative(const Matrix& predicted, const Matrix& target) {
        Matrix result(predicted.rows, predicted.cols);
        double n = static_cast<double>(predicted.data.size());
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            double p = std::clamp(predicted.data[i], EPS, 1.0 - EPS);
            result.data[i] = (-target.data[i] / p + (1.0 - target.data[i]) / (1.0 - p)) / n;
        }
        return result;
    }

    // ====== Categorical Cross-Entropy ======
    // L = -Σ y_true * log(y_pred)
    inline double cross_entropy(const Matrix& predicted, const Matrix& target) {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            double p = std::clamp(predicted.data[i], EPS, 1.0);
            sum += target.data[i] * std::log(p);
        }
        return -sum / static_cast<double>(predicted.rows);
    }

    inline Matrix cross_entropy_derivative(const Matrix& predicted, const Matrix& target) {
        // Softmax + CrossEntropy birlikte: dL/dz = (y_pred - y_true) / batch_size
        Matrix result = predicted - target;
        double n = static_cast<double>(predicted.rows);
        for(size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] /= n;
        }
        return result;
    }

    // ====== Dispatch ======
    inline double compute(Loss type, const Matrix& predicted, const Matrix& target) {
        switch (type) {
            case Loss::MSE:                return mse(predicted, target);
            case Loss::BinaryCrossEntropy: return binary_cross_entropy(predicted, target);
            case Loss::CrossEntropy:       return cross_entropy(predicted, target);
        }
        return 0.0;
    }

    inline Matrix gradient(Loss type, const Matrix& predicted, const Matrix& target) {
        switch (type) {
            case Loss::MSE:                return mse_derivative(predicted, target);
            case Loss::BinaryCrossEntropy: return binary_cross_entropy_derivative(predicted, target);
            case Loss::CrossEntropy:       return cross_entropy_derivative(predicted, target);
        }
        return predicted;
    }

    inline std::string name(Loss type) {
        switch (type) {
            case Loss::MSE:                return "MSE";
            case Loss::BinaryCrossEntropy: return "BinaryCrossEntropy";
            case Loss::CrossEntropy:       return "CrossEntropy";
        }
        return "Unknown";
    }

} // namespace loss
} // namespace xasm
