/*
 * XasmAI — Regression Demo
 * sin(x) fonksiyonunu öğrenen sinir ağı.
 * Evrensel Yaklaşıklama Teoremi'nin pratikte gösterimi.
 *
 * Sinir ağları herhangi bir sürekli fonksiyonu
 * yaklaşık olarak öğrenebilir!
 */

#include "../include/xasmai/xasmai.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

int main() {
    using namespace xasm;

    // Banner göster
    util::banner();

    std::cout << util::color::BOLD << util::color::YELLOW
              << "  🌊 sin(x) Fonksiyon Yaklaşıklama\n"
              << util::color::DIM
              << "  ─────────────────────────────────\n"
              << "  Evrensel Yaklaşıklama Teoremi:\n"
              << "  Bir sinir ağı HERHANGİ bir sürekli\n"
              << "  fonksiyonu öğrenebilir.\n\n"
              << util::color::RESET;

    // ====== Ağ Tanımla ======
    // 1 giriş (x) → 16 → 8 → 1 çıkış (sin(x))
    // Tanh gizli katmanlar + Linear çıkış
    auto net = Network(
        {1, 16, 8, 1},
        {Activation::Tanh, Activation::Tanh, Activation::Linear}
    );
    net.summary("sin(x) Yaklaşıklama Ağı");

    // ====== Eğitim Verisi Üret ======
    // x: [-π, π] arasında 30 nokta
    Dataset data;
    int num_points = 30;

    for (int i = 0; i < num_points; ++i) {
        double x = -M_PI + (2.0 * M_PI * i) / (num_points - 1);
        double y = std::sin(x);
        // Normalize: x → [-1, 1], y zaten [-1, 1]
        double x_norm = x / M_PI;
        data.push_back({{x_norm}, {y}});
    }

    std::cout << util::color::DIM
              << "  📊 " << num_points << " eğitim noktası üretildi (x ∈ [-π, π])\n\n"
              << util::color::RESET;

    // ====== Eğit ======
    net.train(data, {
        .epochs = 3000,
        .lr = 0.003,
        .loss = Loss::MSE,
        .optimizer = OptimizerType::Adam,
        .verbose = true,
        .plot = true
    });

    // ====== Sonuçları Göster ======
    std::cout << util::color::BOLD << util::color::CYAN
              << "\n  ═══════════════════════════════════════\n"
              << "       📋 TAHMİN vs GERÇEK DEĞERLERİ     \n"
              << "  ═══════════════════════════════════════\n\n"
              << util::color::RESET;

    // Test noktaları
    std::vector<double> test_x = {-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0};

    std::cout << util::color::BOLD
              << "  ┌────────────┬────────────┬────────────┬──────────┐\n"
              << "  │     x      │  sin(x)    │  Tahmin    │   Hata   │\n"
              << "  ├────────────┼────────────┼────────────┼──────────┤\n"
              << util::color::RESET;

    double total_error = 0.0;

    for (double x : test_x) {
        double expected = std::sin(x);
        double x_norm = x / M_PI;
        auto result = net.predict({x_norm});
        double predicted = result[0];
        double error = std::abs(predicted - expected);
        total_error += error;

        std::string err_color = error < 0.05 ? util::color::GREEN :
                                error < 0.15 ? util::color::YELLOW : util::color::RED;

        std::cout << "  │ " << std::setw(9) << std::fixed << std::setprecision(4) << x
                  << " │ " << std::setw(9) << expected
                  << " │ " << std::setw(9) << predicted
                  << " │" << err_color << std::setw(8) << error
                  << util::color::RESET << "  │\n";
    }

    std::cout << util::color::BOLD
              << "  └────────────┴────────────┴────────────┴──────────┘\n"
              << util::color::RESET;

    double avg_error = total_error / static_cast<double>(test_x.size());
    std::cout << "\n" << util::color::BOLD
              << "  Ortalama Hata: " << std::fixed << std::setprecision(6) << avg_error << "\n"
              << util::color::RESET;

    // ====== ASCII Grafik: sin(x) vs Tahmin ======
    std::cout << util::color::BOLD << util::color::CYAN
              << "\n  ═══════════════════════════════════════════════════\n"
              << "     📈 sin(x) KARŞILAŞTIRMA GRAFİĞİ                \n"
              << "  ═══════════════════════════════════════════════════\n\n"
              << util::color::RESET;

    int graph_width = 60;
    int graph_height = 15;

    for (int row = graph_height - 1; row >= 0; --row) {
        double y_val = -1.0 + 2.0 * row / (graph_height - 1);

        // Y-axis label
        std::cout << "  " << std::setw(6) << std::fixed << std::setprecision(2) << y_val << " │";

        for (int col = 0; col < graph_width; ++col) {
            double x = -M_PI + (2.0 * M_PI * col) / (graph_width - 1);
            double x_norm = x / M_PI;

            double real_y = std::sin(x);
            auto pred = net.predict({x_norm});
            double pred_y = pred[0];

            bool is_real = std::abs(real_y - y_val) < (2.0 / graph_height);
            bool is_pred = std::abs(pred_y - y_val) < (2.0 / graph_height);

            if (is_real && is_pred)
                std::cout << util::color::GREEN << "◆" << util::color::RESET;
            else if (is_real)
                std::cout << util::color::BLUE << "●" << util::color::RESET;
            else if (is_pred)
                std::cout << util::color::RED << "×" << util::color::RESET;
            else if (row == graph_height / 2)
                std::cout << util::color::DIM << "─" << util::color::RESET;
            else
                std::cout << " ";
        }
        std::cout << "\n";
    }

    std::cout << "         └";
    for (int i = 0; i < graph_width; ++i) std::cout << "─";
    std::cout << "→ x\n";

    std::cout << util::color::DIM
              << "         " << util::color::BLUE << "●" << util::color::DIM << " = sin(x) gerçek   "
              << util::color::RED << "×" << util::color::DIM << " = tahmin   "
              << util::color::GREEN << "◆" << util::color::DIM << " = eşleşme\n\n"
              << util::color::RESET;

    // Modeli kaydet
    net.save("regression_model.xasm");

    return 0;
}
