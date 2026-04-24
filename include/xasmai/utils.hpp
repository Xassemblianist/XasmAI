#pragma once
/*
 * XasmAI — Utility Functions
 * Terminal renklendirme, progress bar, ASCII loss grafiği
 * Dopamin dostu görsel geri bildirim!
 */

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream>

namespace xasm {
namespace util {

    // ====== ANSI Terminal Renkleri ======
    namespace color {
        const std::string RESET   = "\033[0m";
        const std::string BOLD    = "\033[1m";
        const std::string DIM     = "\033[2m";
        const std::string RED     = "\033[31m";
        const std::string GREEN   = "\033[32m";
        const std::string YELLOW  = "\033[33m";
        const std::string BLUE    = "\033[34m";
        const std::string MAGENTA = "\033[35m";
        const std::string CYAN    = "\033[36m";
        const std::string WHITE   = "\033[37m";
        const std::string BG_RED  = "\033[41m";
        const std::string BG_GREEN= "\033[42m";
        const std::string BG_BLUE = "\033[44m";
        const std::string BG_CYAN = "\033[46m";
    }

    // ====== Progress Bar ======
    inline void progress_bar(int current, int total, double loss, double accuracy = -1.0) {
        int bar_width = 30;
        float progress = static_cast<float>(current) / static_cast<float>(total);
        int filled = static_cast<int>(bar_width * progress);

        std::string bar_color;
        if (progress < 0.33)      bar_color = color::RED;
        else if (progress < 0.66) bar_color = color::YELLOW;
        else                      bar_color = color::GREEN;

        std::cout << "\r" << color::BOLD << color::CYAN << "  Epoch "
                  << std::setw(5) << current << "/" << total
                  << color::RESET << " ";

        // Progress bar
        std::cout << color::DIM << "│" << color::RESET;
        std::cout << bar_color;
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled) std::cout << "█";
            else if (i == filled) std::cout << "▓";
            else std::cout << "░";
        }
        std::cout << color::RESET << color::DIM << "│" << color::RESET;

        // Metrics
        std::cout << color::YELLOW << " Loss: " << std::fixed << std::setprecision(6) << loss;

        if (accuracy >= 0.0) {
            std::string acc_color = accuracy > 0.9 ? color::GREEN : color::RED;
            std::cout << acc_color << "  Acc: " << std::fixed << std::setprecision(1)
                      << (accuracy * 100.0) << "%";
        }

        std::cout << color::RESET << "  " << std::flush;
    }

    // ====== ASCII Loss Grafiği ======
    inline void plot_loss(const std::vector<double>& history, int height = 12, int width = 50) {
        if (history.empty()) return;

        double max_loss = *std::max_element(history.begin(), history.end());
        double min_loss = *std::min_element(history.begin(), history.end());
        double range = max_loss - min_loss;
        if (range < 1e-10) range = 1.0;

        std::cout << "\n" << color::BOLD << color::CYAN
                  << "  ╔══════════════════════════════════════════════════════╗\n"
                  << "  ║              📉 LOSS GRAFİĞİ                       ║\n"
                  << "  ╚══════════════════════════════════════════════════════╝\n"
                  << color::RESET;

        // Sample noktaları width'e göre ayarla
        std::vector<double> sampled;
        if (history.size() <= static_cast<size_t>(width)) {
            sampled = history;
        } else {
            for (int i = 0; i < width; ++i) {
                size_t idx = (i * (history.size() - 1)) / (width - 1);
                sampled.push_back(history[idx]);
            }
        }

        int actual_width = static_cast<int>(sampled.size());

        for (int row = height - 1; row >= 0; --row) {
            // Y-axis label
            if (row == height - 1)
                std::cout << color::DIM << "  " << std::setw(8) << std::fixed
                          << std::setprecision(4) << max_loss << " │" << color::RESET;
            else if (row == 0)
                std::cout << color::DIM << "  " << std::setw(8) << std::fixed
                          << std::setprecision(4) << min_loss << " │" << color::RESET;
            else
                std::cout << color::DIM << "           │" << color::RESET;

            for (int col = 0; col < actual_width; ++col) {
                double normalized = (sampled[col] - min_loss) / range * (height - 1);
                if (std::abs(normalized - row) < 0.5) {
                    // Renge göre renk seç
                    std::string c;
                    if (col < actual_width / 3)      c = color::RED;
                    else if (col < 2 * actual_width / 3) c = color::YELLOW;
                    else                              c = color::GREEN;
                    std::cout << c << "●" << color::RESET;
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "\n";
        }

        // X-axis
        std::cout << color::DIM << "           └";
        for (int i = 0; i < actual_width; ++i) std::cout << "─";
        std::cout << "→ epoch\n" << color::RESET;

        // İstatistikler
        std::cout << color::BOLD << "  📊 "
                  << color::GREEN << "Final Loss: " << std::fixed << std::setprecision(6) << history.back()
                  << color::YELLOW << "  |  Başlangıç: " << history.front()
                  << color::CYAN << "  |  Düşüş: " << std::setprecision(1)
                  << ((1.0 - history.back() / history.front()) * 100.0) << "%"
                  << color::RESET << "\n\n";
    }

    // ====== Zamanlayıcı ======
    class Timer {
        std::chrono::high_resolution_clock::time_point start_;
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}

        double elapsed_ms() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_).count();
        }

        double elapsed_sec() const { return elapsed_ms() / 1000.0; }

        std::string elapsed_str() const {
            double ms = elapsed_ms();
            if (ms < 1000.0)
                return std::to_string(static_cast<int>(ms)) + "ms";
            else
                return std::to_string(ms / 1000.0).substr(0, 5) + "s";
        }
    };

    // ====== Banner ======
    inline void banner() {
        std::cout << color::BOLD << color::CYAN << R"(
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║   ██╗  ██╗ █████╗ ███████╗███╗   ███╗     ║
    ║   ╚██╗██╔╝██╔══██╗██╔════╝████╗ ████║     ║
    ║    ╚███╔╝ ███████║███████╗██╔████╔██║     ║
    ║    ██╔██╗ ██╔══██║╚════██║██║╚██╔╝██║     ║
    ║   ██╔╝ ██╗██║  ██║███████║██║ ╚═╝ ██║     ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝     ║
    ║              A I   E N G I N E              ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
)" << color::RESET;
        std::cout << color::DIM << "    C++20 Neural Network Framework — v1.0\n"
                  << "    No dependencies. Pure speed.\n\n" << color::RESET;
    }

    // ====== Model Özet Tablosu ======
    inline void print_summary(const std::string& title, const std::vector<std::string>& layer_infos, size_t total_params) {
        std::cout << color::BOLD << color::MAGENTA
                  << "\n  ┌──────────────────────────────────────────────┐\n"
                  << "  │  🧠 " << title << std::string(42 - title.size(), ' ') << "│\n"
                  << "  ├──────────────────────────────────────────────┤\n" << color::RESET;

        for (size_t i = 0; i < layer_infos.size(); ++i) {
            std::string info = layer_infos[i];
            int padding = 44 - static_cast<int>(info.size());
            if (padding < 0) padding = 0;
            std::cout << color::WHITE << "  │  Layer " << i << ": " << info
                      << std::string(padding - 10, ' ') << "│\n" << color::RESET;
        }

        std::cout << color::BOLD << color::MAGENTA
                  << "  ├──────────────────────────────────────────────┤\n"
                  << color::YELLOW
                  << "  │  Toplam Parametre: " << total_params
                  << std::string(25 - std::to_string(total_params).size(), ' ') << "│\n"
                  << color::MAGENTA
                  << "  └──────────────────────────────────────────────┘\n\n"
                  << color::RESET;
    }

} // namespace util
} // namespace xasm
