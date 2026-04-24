#include "../include/xasmai/xasmai.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    using namespace xasm;

    constexpr size_t LATENT_DIM   = 16;
    constexpr size_t DATA_DIM     = 8;
    constexpr size_t G_HIDDEN     = 32;
    constexpr size_t D_HIDDEN     = 32;
    constexpr double G_LR         = 0.0002;
    constexpr double D_LR         = 0.0002;
    constexpr int    LOG_EVERY    = 100;

    Network generator(
        {LATENT_DIM, G_HIDDEN, G_HIDDEN, DATA_DIM},
        {Activation::LeakyReLU, Activation::LeakyReLU, Activation::Tanh}
    );

    Network discriminator(
        {DATA_DIM, D_HIDDEN, D_HIDDEN, 1},
        {Activation::LeakyReLU, Activation::LeakyReLU, Activation::Sigmoid}
    );

    Optimizer g_opt(OptimizerConfig{.type = OptimizerType::Adam, .lr = G_LR, .beta1 = 0.5, .beta2 = 0.999});
    Optimizer d_opt(OptimizerConfig{.type = OptimizerType::Adam, .lr = D_LR, .beta1 = 0.5, .beta2 = 0.999});

    generator.optimizer = g_opt;
    discriminator.optimizer = d_opt;

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> noise_dist(0.0, 1.0);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);

    auto sample_noise = [&]() -> Matrix {
        Matrix z(1, LATENT_DIM);
        for (auto& v : z.data) v = noise_dist(rng);
        return z;
    };

    auto sample_real = [&]() -> Matrix {
        Matrix x(1, DATA_DIM);
        double phase = phase_dist(rng);
        double freq = 1.0 + noise_dist(rng) * 0.1;
        for (size_t i = 0; i < DATA_DIM; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(DATA_DIM - 1);
            x(0, i) = std::sin(freq * t * M_PI * 2.0 + phase) * 0.8 + noise_dist(rng) * 0.05;
        }
        return x;
    };

    std::cout << "\033[2J\033[H";
    std::cout << util::color::BOLD << util::color::RED << R"(
     ██████╗  █████╗ ███╗   ██╗
    ██╔════╝ ██╔══██╗████╗  ██║
    ██║  ███╗███████║██╔██╗ ██║
    ██║   ██║██╔══██║██║╚██╗██║
    ╚██████╔╝██║  ██║██║ ╚████║
     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝
)" << util::color::RESET;

    std::cout << util::color::DIM
              << "    Generative Adversarial Network\n"
              << "    Generator vs Discriminator\n"
              << "    Latent: " << LATENT_DIM << " | Data: " << DATA_DIM << "\n"
              << "    ────────────────────────────────\n\n" << util::color::RESET;

    size_t total_g = generator.total_params();
    size_t total_d = discriminator.total_params();
    std::cout << util::color::CYAN << "    G params: " << total_g
              << "  |  D params: " << total_d
              << "  |  Total: " << (total_g + total_d)
              << util::color::RESET << "\n\n";

    std::cout << util::color::BOLD
              << "  ┌──────────┬────────────┬────────────┬────────────┬──────────┐\n"
              << "  │   Iter   │   D_Loss   │   G_Loss   │  D(real)   │  iter/s  │\n"
              << "  ├──────────┼────────────┼────────────┼────────────┼──────────┤\n"
              << util::color::RESET;

    auto wall_start = std::chrono::high_resolution_clock::now();
    double d_loss_avg = 0.0, g_loss_avg = 0.0;
    double d_real_avg = 0.0;
    int avg_count = 0;

    for (size_t iter = 1; ; ++iter) {

        Matrix real_data = sample_real();
        Matrix real_out = discriminator.forward(real_data);
        Matrix real_target(1, 1, {1.0});
        double d_loss_real = loss::binary_cross_entropy(real_out, real_target);
        Matrix d_grad_real = loss::binary_cross_entropy_derivative(real_out, real_target);
        discriminator.backward(d_grad_real);
        discriminator.update_weights();

        Matrix z = sample_noise();
        Matrix fake_data = generator.forward(z);
        Matrix fake_out = discriminator.forward(fake_data);
        Matrix fake_target(1, 1, {0.0});
        double d_loss_fake = loss::binary_cross_entropy(fake_out, fake_target);
        Matrix d_grad_fake = loss::binary_cross_entropy_derivative(fake_out, fake_target);
        discriminator.backward(d_grad_fake);
        discriminator.update_weights();

        double d_loss = (d_loss_real + d_loss_fake) * 0.5;

        Matrix z2 = sample_noise();
        Matrix gen_data = generator.forward(z2);
        Matrix gen_out = discriminator.forward(gen_data);
        Matrix gen_target(1, 1, {1.0});
        double g_loss_val = loss::binary_cross_entropy(gen_out, gen_target);

        Matrix g_d_grad = loss::binary_cross_entropy_derivative(gen_out, gen_target);
        Matrix g_through_d = discriminator.backward(g_d_grad);
        generator.backward(g_through_d);
        generator.update_weights();

        d_loss_avg += d_loss;
        g_loss_avg += g_loss_val;
        d_real_avg += real_out(0, 0);
        avg_count++;

        if (iter % LOG_EVERY == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            double ips = static_cast<double>(iter) / elapsed;

            double dl = d_loss_avg / avg_count;
            double gl = g_loss_avg / avg_count;
            double dr = d_real_avg / avg_count;

            std::string d_color, g_color;
            if (dl < 0.5) d_color = util::color::GREEN;
            else if (dl < 1.0) d_color = util::color::YELLOW;
            else d_color = util::color::RED;

            if (gl < 1.0) g_color = util::color::GREEN;
            else if (gl < 2.0) g_color = util::color::YELLOW;
            else g_color = util::color::RED;

            std::string eq_color;
            if (std::abs(dl - gl) < 0.3) eq_color = util::color::GREEN;
            else eq_color = util::color::RED;

            std::cout << "  │ " << std::setw(8) << iter << " │"
                      << d_color << std::setw(11) << std::fixed << std::setprecision(6) << dl << util::color::RESET << " │"
                      << g_color << std::setw(11) << std::fixed << std::setprecision(6) << gl << util::color::RESET << " │"
                      << util::color::CYAN << std::setw(11) << std::fixed << std::setprecision(6) << dr << util::color::RESET << " │"
                      << util::color::DIM << std::setw(8) << std::fixed << std::setprecision(1) << ips << util::color::RESET << "  │"
                      << eq_color;

            double balance = 1.0 - std::abs(dl - gl) / std::max(dl + gl, 0.001);
            int bar_len = static_cast<int>(balance * 10);
            for (int b = 0; b < 10; ++b)
                std::cout << (b < bar_len ? "█" : "░");

            std::cout << util::color::RESET << "\n";

            d_loss_avg = 0.0;
            g_loss_avg = 0.0;
            d_real_avg = 0.0;
            avg_count = 0;

            if (iter % (LOG_EVERY * 20) == 0) {
                std::cout << util::color::DIM
                          << "  ├──────────┼────────────┼────────────┼────────────┼──────────┤\n"
                          << util::color::RESET;

                std::cout << util::color::BOLD << util::color::MAGENTA
                          << "  │ Sample G: [";
                Matrix sample = generator.forward(sample_noise());
                for (size_t i = 0; i < sample.cols; ++i) {
                    double v = sample(0, i);
                    if (v > 0.5) std::cout << util::color::GREEN << "▓";
                    else if (v > 0.0) std::cout << util::color::YELLOW << "▒";
                    else if (v > -0.5) std::cout << util::color::CYAN << "░";
                    else std::cout << util::color::BLUE << " ";
                }
                std::cout << util::color::MAGENTA << "]"
                          << std::string(36 - sample.cols, ' ') << "│\n";

                std::cout << util::color::DIM
                          << "  ├──────────┼────────────┼────────────┼────────────┼──────────┤\n"
                          << util::color::RESET;
            }
        }
    }

    return 0;
}
