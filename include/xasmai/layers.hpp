#pragma once

#include "matrix.hpp"
#include "activations.hpp"
#include "optimizer.hpp"
#include <random>

namespace xasm {

class DenseLayer {
public:
    Matrix weights;
    Matrix biases;
    Activation act_type;

    Matrix input_cache;
    Matrix z_cache;
    Matrix output_cache;

    Matrix d_weights;
    Matrix d_biases;

    ParamState weight_state;
    ParamState bias_state;

    DenseLayer() = default;

    DenseLayer(size_t input_size, size_t output_size, Activation activation = Activation::ReLU)
        : act_type(activation) {
        if (activation == Activation::ReLU || activation == Activation::LeakyReLU) {
            weights = Matrix::he(input_size, output_size);
        } else {
            weights = Matrix::xavier(input_size, output_size);
        }
        biases = Matrix::zeros(1, output_size);
        weight_state.init(input_size, output_size);
        bias_state.init(1, output_size);
    }

    Matrix forward(const Matrix& input) {
        input_cache = input;
        Matrix inp = input;
        if (input.cols == 1 && input.rows > 1)
            inp = input.T();

        z_cache = inp * weights;
        for (size_t i = 0; i < z_cache.rows; ++i) {
            for (size_t j = 0; j < z_cache.cols; ++j) {
                z_cache(i, j) += biases(0, j);
            }
        }

        output_cache = act::forward(act_type, z_cache);
        return output_cache;
    }

    Matrix backward(const Matrix& d_output) {
        Matrix d_act;
        if (act_type == Activation::Softmax) {
            d_act = d_output;
        } else {
            Matrix act_deriv = act::derivative(act_type, output_cache);
            d_act = d_output.hadamard(act_deriv);
        }

        Matrix inp = input_cache;
        if (inp.cols == 1 && inp.rows > 1)
            inp = inp.T();

        d_weights = inp.T() * d_act;
        d_biases = d_act.sum_rows();

        Matrix d_input = d_act * weights.T();
        return d_input;
    }

    void update_weights(Optimizer& opt) {
        opt.update(weights, d_weights, weight_state);
        opt.update(biases, d_biases, bias_state);
    }

    size_t input_size() const { return weights.rows; }
    size_t output_size() const { return weights.cols; }
    size_t param_count() const { return weights.size() + biases.size(); }

    std::string info() const {
        return "Dense(" + std::to_string(input_size()) + " -> " +
               std::to_string(output_size()) + ", " +
               act::name(act_type) + ") [" +
               std::to_string(param_count()) + " params]";
    }
};

class DropoutLayer {
    double rate;
    Matrix mask;
    std::mt19937 rng;
    bool training = true;

public:
    DropoutLayer() : rate(0.5), rng(std::random_device{}()) {}

    explicit DropoutLayer(double dropout_rate)
        : rate(dropout_rate), rng(std::random_device{}()) {}

    void set_training(bool mode) { training = mode; }

    Matrix forward(const Matrix& input) {
        if (!training)
            return input;

        mask = Matrix(input.rows, input.cols);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double scale = 1.0 / (1.0 - rate);

        for (size_t i = 0; i < mask.data.size(); ++i) {
            mask.data[i] = dist(rng) > rate ? scale : 0.0;
        }

        return input.hadamard(mask);
    }

    Matrix backward(const Matrix& d_output) {
        if (!training)
            return d_output;
        return d_output.hadamard(mask);
    }

    size_t param_count() const { return 0; }

    std::string info() const {
        return "Dropout(rate=" + std::to_string(rate) + ")";
    }
};

class BatchNormLayer {
    Matrix gamma;
    Matrix beta;
    Matrix running_mean;
    Matrix running_var;

    Matrix x_norm;
    Matrix x_centered;
    Matrix inv_std;
    Matrix input_cache;

    ParamState gamma_state;
    ParamState beta_state;

    Matrix d_gamma;
    Matrix d_beta;

    double momentum;
    double epsilon;
    bool training = true;
    size_t features;

public:
    BatchNormLayer() : momentum(0.1), epsilon(1e-5), features(0) {}

    explicit BatchNormLayer(size_t num_features, double mom = 0.1, double eps = 1e-5)
        : momentum(mom), epsilon(eps), features(num_features) {
        gamma = Matrix::ones(1, features);
        beta = Matrix::zeros(1, features);
        running_mean = Matrix::zeros(1, features);
        running_var = Matrix::ones(1, features);
        gamma_state.init(1, features);
        beta_state.init(1, features);
    }

    void set_training(bool mode) { training = mode; }

    Matrix forward(const Matrix& input) {
        input_cache = input;

        Matrix mean_v(1, input.cols, 0.0);
        Matrix var_v(1, input.cols, 0.0);

        if (training) {
            for (size_t j = 0; j < input.cols; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < input.rows; ++i)
                    sum += input(i, j);
                mean_v(0, j) = sum / static_cast<double>(input.rows);
            }

            for (size_t j = 0; j < input.cols; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < input.rows; ++i) {
                    double d = input(i, j) - mean_v(0, j);
                    sum += d * d;
                }
                var_v(0, j) = sum / static_cast<double>(input.rows);
            }

            for (size_t j = 0; j < input.cols; ++j) {
                running_mean(0, j) = (1.0 - momentum) * running_mean(0, j) + momentum * mean_v(0, j);
                running_var(0, j) = (1.0 - momentum) * running_var(0, j) + momentum * var_v(0, j);
            }
        } else {
            mean_v = running_mean;
            var_v = running_var;
        }

        inv_std = Matrix(1, input.cols);
        for (size_t j = 0; j < input.cols; ++j)
            inv_std(0, j) = 1.0 / std::sqrt(var_v(0, j) + epsilon);

        x_centered = Matrix(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                x_centered(i, j) = input(i, j) - mean_v(0, j);

        x_norm = Matrix(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                x_norm(i, j) = x_centered(i, j) * inv_std(0, j);

        Matrix output(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                output(i, j) = gamma(0, j) * x_norm(i, j) + beta(0, j);

        return output;
    }

    Matrix backward(const Matrix& d_output) {
        double N = static_cast<double>(d_output.rows);

        d_gamma = Matrix::zeros(1, d_output.cols);
        d_beta = Matrix::zeros(1, d_output.cols);

        for (size_t i = 0; i < d_output.rows; ++i) {
            for (size_t j = 0; j < d_output.cols; ++j) {
                d_gamma(0, j) += d_output(i, j) * x_norm(i, j);
                d_beta(0, j) += d_output(i, j);
            }
        }

        Matrix dx_norm(d_output.rows, d_output.cols);
        for (size_t i = 0; i < d_output.rows; ++i)
            for (size_t j = 0; j < d_output.cols; ++j)
                dx_norm(i, j) = d_output(i, j) * gamma(0, j);

        Matrix d_input(d_output.rows, d_output.cols);
        for (size_t j = 0; j < d_output.cols; ++j) {
            double sum_dx = 0.0, sum_dx_xc = 0.0;
            for (size_t i = 0; i < d_output.rows; ++i) {
                sum_dx += dx_norm(i, j);
                sum_dx_xc += dx_norm(i, j) * x_centered(i, j);
            }

            for (size_t i = 0; i < d_output.rows; ++i) {
                d_input(i, j) = inv_std(0, j) / N * (
                    N * dx_norm(i, j) - sum_dx -
                    x_centered(i, j) * inv_std(0, j) * inv_std(0, j) * sum_dx_xc
                );
            }
        }

        return d_input;
    }

    void update_weights(Optimizer& opt) {
        opt.update(gamma, d_gamma, gamma_state);
        opt.update(beta, d_beta, beta_state);
    }

    size_t param_count() const { return features * 2; }

    std::string info() const {
        return "BatchNorm(" + std::to_string(features) + ") [" +
               std::to_string(param_count()) + " params]";
    }
};

} // namespace xasm
