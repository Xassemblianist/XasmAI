#pragma once

#include "xasmai.hpp"
#include <deque>
#include <random>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <functional>

namespace xasm {

struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

class ReplayBuffer {
    std::deque<Experience> buffer;
    size_t capacity;
    std::mt19937 rng;

public:
    explicit ReplayBuffer(size_t cap = 100000)
        : capacity(cap), rng(std::random_device{}()) {}

    void push(Experience exp) {
        if (buffer.size() >= capacity)
            buffer.pop_front();
        buffer.push_back(std::move(exp));
    }

    std::vector<Experience> sample(size_t batch_size) {
        std::vector<Experience> batch;
        batch.reserve(batch_size);
        size_t n = std::min(batch_size, buffer.size());
        std::vector<size_t> indices(buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < n; ++i)
            batch.push_back(buffer[indices[i]]);
        return batch;
    }

    size_t size() const { return buffer.size(); }
    void clear() { buffer.clear(); }
};

struct AgentConfig {
    std::vector<size_t> topology;
    double epsilon = 1.0;
    double epsilon_min = 0.01;
    double epsilon_decay = 0.995;
    double gamma = 0.99;
    double lr = 0.001;
    size_t replay_capacity = 100000;
    size_t batch_size = 64;
    size_t target_update_freq = 100;
    OptimizerType optimizer = OptimizerType::Adam;
};

class Agent {
    Network q_net;
    Network target_net;
    AgentConfig cfg;
    std::mt19937 rng;
    ReplayBuffer memory;
    size_t step_count = 0;
    size_t action_space;
    double total_reward = 0.0;
    size_t episode_count = 0;

    void sync_target() {
        target_net.layers.clear();
        for (const auto& layer : q_net.layers) {
            DenseLayer copy(layer.input_size(), layer.output_size(), layer.act_type);
            copy.weights = layer.weights;
            copy.biases = layer.biases;
            target_net.layers.push_back(std::move(copy));
        }
    }

    std::vector<double> q_values(Network& net, const std::vector<double>& state) {
        return net.predict(state);
    }

    int argmax(const std::vector<double>& v) {
        return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
    }

public:
    Agent() = default;

    explicit Agent(AgentConfig config)
        : cfg(config), rng(std::random_device{}()),
          memory(config.replay_capacity) {

        action_space = config.topology.back();

        std::vector<Activation> activations;
        for (size_t i = 0; i < config.topology.size() - 1; ++i) {
            if (i == config.topology.size() - 2)
                activations.push_back(Activation::Linear);
            else
                activations.push_back(Activation::ReLU);
        }

        q_net = Network(config.topology, activations);
        sync_target();
    }

    int get_action(const std::vector<double>& state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < cfg.epsilon) {
            std::uniform_int_distribution<int> act_dist(0, static_cast<int>(action_space) - 1);
            return act_dist(rng);
        }
        auto qv = q_values(q_net, state);
        return argmax(qv);
    }

    int get_best_action(const std::vector<double>& state) {
        auto qv = q_values(q_net, state);
        return argmax(qv);
    }

    void remember(const std::vector<double>& state, int action, double reward,
                  const std::vector<double>& next_state, bool done) {
        memory.push({state, action, reward, next_state, done});
    }

    double replay() {
        return replay(cfg.batch_size);
    }

    double replay(size_t batch_size) {
        if (memory.size() < batch_size)
            return 0.0;

        auto batch = memory.sample(batch_size);

        q_net.optimizer = Optimizer(OptimizerConfig{
            .type = cfg.optimizer,
            .lr = cfg.lr
        });

        double total_loss = 0.0;

        for (const auto& exp : batch) {
            auto current_q = q_values(q_net, exp.state);

            double target_val;
            if (exp.done) {
                target_val = exp.reward;
            } else {
                auto next_q = q_values(target_net, exp.next_state);
                target_val = exp.reward + cfg.gamma * *std::max_element(next_q.begin(), next_q.end());
            }

            auto target_q = current_q;
            target_q[exp.action] = target_val;

            Matrix inp = Matrix::row_vector(exp.state);
            Matrix tgt = Matrix::row_vector(target_q);
            Matrix output = q_net.forward(inp);

            double sample_loss = loss::mse(output, tgt);
            total_loss += sample_loss;

            Matrix grad = loss::mse_derivative(output, tgt);
            q_net.backward(grad);
            q_net.update_weights();
        }

        step_count++;
        if (step_count % cfg.target_update_freq == 0)
            sync_target();

        decay_epsilon();

        return total_loss / static_cast<double>(batch_size);
    }

    void decay_epsilon() {
        if (cfg.epsilon > cfg.epsilon_min)
            cfg.epsilon *= cfg.epsilon_decay;
    }

    void new_episode(double episode_reward = 0.0) {
        episode_count++;
        total_reward += episode_reward;
    }

    double get_epsilon() const { return cfg.epsilon; }
    size_t get_step_count() const { return step_count; }
    size_t get_episode_count() const { return episode_count; }
    size_t get_memory_size() const { return memory.size(); }
    double get_avg_reward() const {
        return episode_count > 0 ? total_reward / static_cast<double>(episode_count) : 0.0;
    }

    void set_epsilon(double eps) { cfg.epsilon = eps; }
    void set_lr(double lr) { cfg.lr = lr; }

    void save(const std::string& path) { q_net.save(path); }

    void load(const std::string& path) {
        q_net = Network::load(path);
        sync_target();
    }

    void summary() const { q_net.summary("DQN Agent"); }

    Network& network() { return q_net; }
    const AgentConfig& config() const { return cfg; }

    void print_status() const {
        std::cout << util::color::BOLD << util::color::CYAN
                  << "  ┌───────────────────────────────────┐\n"
                  << "  │        AJAN DURUM PANELİ          │\n"
                  << "  ├───────────────────────────────────┤\n"
                  << util::color::RESET
                  << "  │ Episode:  " << std::setw(8) << episode_count << "              │\n"
                  << "  │ Step:     " << std::setw(8) << step_count << "              │\n"
                  << "  │ Epsilon:  " << std::setw(8) << std::fixed << std::setprecision(4) << cfg.epsilon << "              │\n"
                  << "  │ Hafıza:   " << std::setw(8) << memory.size() << "              │\n"
                  << "  │ Ort.Ödül: " << std::setw(8) << std::fixed << std::setprecision(2) << get_avg_reward() << "              │\n"
                  << util::color::BOLD << util::color::CYAN
                  << "  └───────────────────────────────────┘\n"
                  << util::color::RESET;
    }
};

} // namespace xasm
