#pragma once

#include "matrix.hpp"
#include <cmath>
#include <string>
#include <fstream>

namespace xasm {

enum class OptimizerType {
    SGD,
    Momentum,
    Adam,
    AdamW
};

struct OptimizerConfig {
    OptimizerType type = OptimizerType::AdamW;
    double lr = 3e-4;
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.01;
};

struct ParamState {
    Matrix velocity;
    Matrix cache;
    int t = 0;

    ParamState() = default;

    void init(size_t rows, size_t cols) {
        velocity = Matrix::zeros(rows, cols);
        cache = Matrix::zeros(rows, cols);
        t = 0;
    }

    void serialize(std::ostream& os) const {
        velocity.serialize(os);
        cache.serialize(os);
        int32_t tt = t;
        os.write(reinterpret_cast<const char*>(&tt), 4);
    }

    void deserialize(std::istream& is) {
        velocity = Matrix::deserialize(is);
        cache = Matrix::deserialize(is);
        int32_t tt;
        is.read(reinterpret_cast<char*>(&tt), 4);
        t = tt;
    }
};

class LRScheduler {
    double base_lr_;
    double min_lr_;
    size_t warmup_steps_;
    size_t total_steps_;
    size_t current_step_;

public:
    LRScheduler()
        : base_lr_(3e-4), min_lr_(1e-5),
          warmup_steps_(1000), total_steps_(1000000),
          current_step_(0) {}

    LRScheduler(double base_lr, double min_lr,
                size_t warmup, size_t total)
        : base_lr_(base_lr), min_lr_(min_lr),
          warmup_steps_(warmup), total_steps_(total),
          current_step_(0) {}

    double step() {
        current_step_++;
        return get_lr();
    }

    double get_lr() const {
        if (current_step_ <= warmup_steps_) {
            return base_lr_ * static_cast<double>(current_step_) /
                   static_cast<double>(std::max(warmup_steps_, size_t(1)));
        }
        double progress = static_cast<double>(current_step_ - warmup_steps_) /
                           static_cast<double>(std::max(total_steps_ - warmup_steps_, size_t(1)));
        progress = std::min(progress, 1.0);
        return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + std::cos(progress * M_PI));
    }

    void set_step(size_t s) { current_step_ = s; }
    size_t get_step() const { return current_step_; }

    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&base_lr_), 8);
        os.write(reinterpret_cast<const char*>(&min_lr_), 8);
        uint64_t w = warmup_steps_, t = total_steps_, c = current_step_;
        os.write(reinterpret_cast<const char*>(&w), 8);
        os.write(reinterpret_cast<const char*>(&t), 8);
        os.write(reinterpret_cast<const char*>(&c), 8);
    }

    void deserialize(std::istream& is) {
        is.read(reinterpret_cast<char*>(&base_lr_), 8);
        is.read(reinterpret_cast<char*>(&min_lr_), 8);
        uint64_t w, t, c;
        is.read(reinterpret_cast<char*>(&w), 8);
        is.read(reinterpret_cast<char*>(&t), 8);
        is.read(reinterpret_cast<char*>(&c), 8);
        warmup_steps_ = w; total_steps_ = t; current_step_ = c;
    }
};

class Optimizer {
public:
    OptimizerConfig config;

    Optimizer() = default;
    explicit Optimizer(OptimizerConfig cfg) : config(cfg) {}
    explicit Optimizer(double lr) : config{.lr = lr} {}

    void set_lr(double lr) { config.lr = lr; }

    void update(Matrix& param, const Matrix& grad, ParamState& state) {
        switch (config.type) {
            case OptimizerType::SGD:
                sgd_update(param, grad);
                break;
            case OptimizerType::Momentum:
                momentum_update(param, grad, state);
                break;
            case OptimizerType::Adam:
                adam_update(param, grad, state, false);
                break;
            case OptimizerType::AdamW:
                adam_update(param, grad, state, true);
                break;
        }
    }

    std::string name() const {
        switch (config.type) {
            case OptimizerType::SGD:      return "SGD";
            case OptimizerType::Momentum: return "SGD+Momentum";
            case OptimizerType::Adam:     return "Adam";
            case OptimizerType::AdamW:    return "AdamW";
        }
        return "Unknown";
    }

private:
    void sgd_update(Matrix& param, const Matrix& grad) {
        param -= grad * config.lr;
    }

    void momentum_update(Matrix& param, const Matrix& grad, ParamState& state) {
        if (state.velocity.size() == 0)
            state.init(param.rows, param.cols);
        state.velocity = state.velocity * config.momentum - grad * config.lr;
        param += state.velocity;
    }

    void adam_update(Matrix& param, const Matrix& grad, ParamState& state, bool decoupled_wd) {
        if (state.velocity.size() == 0)
            state.init(param.rows, param.cols);

        state.t++;
        double t = static_cast<double>(state.t);

        state.velocity = state.velocity * config.beta1 + grad * (1.0 - config.beta1);

        Matrix grad_sq = grad.hadamard(grad);
        state.cache = state.cache * config.beta2 + grad_sq * (1.0 - config.beta2);

        Matrix m_hat = state.velocity / (1.0 - std::pow(config.beta1, t));
        Matrix v_hat = state.cache / (1.0 - std::pow(config.beta2, t));

        Matrix step(param.rows, param.cols);
        for (size_t i = 0; i < param.data.size(); ++i)
            step.data[i] = config.lr * m_hat.data[i] / (std::sqrt(v_hat.data[i]) + config.epsilon);

        if (decoupled_wd && config.weight_decay > 0.0) {
            for (size_t i = 0; i < param.data.size(); ++i)
                param.data[i] *= (1.0 - config.lr * config.weight_decay);
        }

        param -= step;
    }
};

} // namespace xasm
