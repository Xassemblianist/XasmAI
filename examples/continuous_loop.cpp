#include "../include/xasmai/agent.hpp"
#include <iostream>
#include <vector>
#include <functional>

struct Environment {
    virtual std::vector<double> reset() = 0;
    virtual std::tuple<std::vector<double>, double, bool> step(int action) = 0;
    virtual size_t state_size() const = 0;
    virtual size_t action_size() const = 0;
    virtual ~Environment() = default;
};

int main() {
    using namespace xasm;

    util::banner();

    // ================================================================
    //  BU BLOĞU KENDİ ENVIRONMENT'INLA DEĞİŞTİR
    //  Aşağıdaki satırları kendi state/action boyutlarınla güncelle
    // ================================================================

    size_t STATE_DIM  = 4;
    size_t ACTION_DIM = 2;

    Agent agent(AgentConfig{
        .topology = {STATE_DIM, 64, 32, ACTION_DIM},
        .epsilon = 1.0,
        .epsilon_min = 0.01,
        .epsilon_decay = 0.995,
        .gamma = 0.99,
        .lr = 0.001,
        .replay_capacity = 50000,
        .batch_size = 32,
        .target_update_freq = 50,
        .optimizer = OptimizerType::Adam
    });

    agent.summary();

    size_t max_episodes = 10000;
    size_t max_steps_per_episode = 500;
    size_t warmup = 100;

    std::vector<double> reward_history;

    for (size_t episode = 0; episode < max_episodes; ++episode) {

        // ============================================================
        //  state = env.reset();
        //  Kendi environment'ının reset fonksiyonunu buraya yaz
        // ============================================================
        std::vector<double> state(STATE_DIM, 0.0);

        double episode_reward = 0.0;
        bool done = false;

        for (size_t step = 0; step < max_steps_per_episode && !done; ++step) {

            int action = agent.get_action(state);

            // ============================================================
            //  auto [next_state, reward, done] = env.step(action);
            //  Kendi environment'ının step fonksiyonunu buraya yaz
            // ============================================================
            std::vector<double> next_state(STATE_DIM, 0.0);
            double reward = 0.0;
            done = false;

            agent.remember(state, action, reward, next_state, done);

            if (agent.get_memory_size() > warmup) {
                double loss = agent.replay();
                (void)loss;
            }

            state = next_state;
            episode_reward += reward;
        }

        agent.new_episode(episode_reward);
        reward_history.push_back(episode_reward);

        if ((episode + 1) % 50 == 0) {
            double avg_50 = 0.0;
            size_t start = reward_history.size() > 50 ? reward_history.size() - 50 : 0;
            for (size_t i = start; i < reward_history.size(); ++i)
                avg_50 += reward_history[i];
            avg_50 /= static_cast<double>(reward_history.size() - start);

            std::cout << util::color::BOLD << util::color::CYAN
                      << "  Episode " << std::setw(5) << (episode + 1)
                      << util::color::YELLOW
                      << "  │  Ödül: " << std::setw(8) << std::fixed << std::setprecision(2) << episode_reward
                      << util::color::GREEN
                      << "  │  Ort(50): " << std::setw(8) << avg_50
                      << util::color::MAGENTA
                      << "  │  ε: " << std::setw(6) << std::setprecision(4) << agent.get_epsilon()
                      << util::color::DIM
                      << "  │  Hafıza: " << agent.get_memory_size()
                      << util::color::RESET << "\n";
        }

        if ((episode + 1) % 500 == 0) {
            agent.print_status();
            agent.save("agent_checkpoint.xasm");
        }
    }

    std::cout << "\n";
    agent.print_status();
    util::plot_loss(reward_history, 12, 50);
    agent.save("agent_final.xasm");

    return 0;
}
