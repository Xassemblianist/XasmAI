/*
 * XasmAI — XOR Demo
 * XOR mantık kapısını öğrenen sinir ağı.
 * En temel neural network testi.
 *
 * XOR Doğruluk Tablosu:
 *   0 XOR 0 = 0
 *   0 XOR 1 = 1
 *   1 XOR 0 = 1
 *   1 XOR 1 = 0
 *
 * Bu problemi tek bir nöron çözemez (lineer ayrılamaz)
 * → en az bir gizli katman gerekir!
 */

#include "../include/xasmai/xasmai.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace xasm;

    // Banner göster
    util::banner();

    std::cout << util::color::BOLD << util::color::YELLOW
              << "  🔬 XOR Problem Çözücü\n"
              << util::color::DIM
              << "  ─────────────────────────────────\n"
              << "  Doğrusal olarak ayrılamayan problemi\n"
              << "  sinir ağı ile çözüyoruz.\n\n"
              << util::color::RESET;

    // ====== Ağ Tanımla ======
    // 2 giriş → 16 gizli → 8 gizli → 1 çıkış
    auto net = Network({2, 16, 8, 1});
    net.summary("XOR Çözücü Ağ");

    // ====== Eğitim Verisi ======
    Dataset data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    // ====== Eğit ======
    net.train(data, {
        .epochs = 5000,
        .lr = 0.01,
        .loss = Loss::MSE,
        .optimizer = OptimizerType::Adam,
        .verbose = true,
        .plot = true
    });

    // ====== Sonuçları Göster ======
    std::cout << util::color::BOLD << util::color::CYAN
              << "\n  ═══════════════════════════════════\n"
              << "         📋 TAHMİN SONUÇLARI         \n"
              << "  ═══════════════════════════════════\n\n"
              << util::color::RESET;

    std::cout << util::color::BOLD
              << "  ┌─────────┬─────────┬──────────┬────────┐\n"
              << "  │  Giriş  │  Hedef  │  Tahmin  │ Durum  │\n"
              << "  ├─────────┼─────────┼──────────┼────────┤\n"
              << util::color::RESET;

    int correct = 0;
    for (const auto& [input, target] : data) {
        auto result = net.predict(input);
        double predicted = result[0];
        double expected = target[0];
        bool is_correct = std::abs(std::round(predicted) - expected) < 0.5;
        if (is_correct) correct++;

        std::string status = is_correct ?
            util::color::GREEN + "  ✅  " + util::color::RESET :
            util::color::RED + "  ❌  " + util::color::RESET;

        std::cout << "  │ " << std::fixed << std::setprecision(0)
                  << input[0] << ", " << input[1]
                  << "   │   " << expected
                  << "     │ " << std::setprecision(4) << predicted
                  << "  │" << status << "│\n";
    }

    std::cout << util::color::BOLD
              << "  └─────────┴─────────┴──────────┴────────┘\n"
              << util::color::RESET;

    std::cout << "\n" << util::color::BOLD
              << "  Doğruluk: " << correct << "/4 ("
              << (correct * 100 / 4) << "%)\n\n"
              << util::color::RESET;

    // ====== Modeli Kaydet ======
    net.save("xor_model.xasm");

    // ====== Modeli Geri Yükle ======
    std::cout << "\n  📂 Model geri yükleniyor...\n";
    auto loaded = Network::load("xor_model.xasm");
    auto test = loaded.predict({1, 0});
    std::cout << "  🔄 Yüklenen model testi: predict({1, 0}) = "
              << std::fixed << std::setprecision(4) << test[0]
              << " (beklenen: ~1.0)\n\n";

    return 0;
}
