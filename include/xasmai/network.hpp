#pragma once
/*
 * XasmAI — Neural Network
 * Ana ağ sınıfı: katmanlar, eğitim, tahmin, kaydet/yükle
 */

#include "matrix.hpp"
#include "layers.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "activations.hpp"
#include "utils.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <functional>

namespace xasm {

// Eğitim verisi — {girdi, hedef} çiftleri
using Sample = std::pair<std::vector<double>, std::vector<double>>;
using Dataset = std::vector<Sample>;

class DynamicLossScaler {
    double scale_;
    double growth_factor_;
    double backoff_factor_;
    int growth_interval_;
    int steps_ok_;
    int overflow_count_;
    int total_steps_;

public:
    DynamicLossScaler(double init_scale = 65536.0,
                       double growth = 2.0,
                       double backoff = 0.5,
                       int interval = 2000)
        : scale_(init_scale), growth_factor_(growth),
          backoff_factor_(backoff), growth_interval_(interval),
          steps_ok_(0), overflow_count_(0), total_steps_(0) {}

    double scale() const { return scale_; }

    Matrix scale_loss(const Matrix& grad) const {
        return grad * scale_;
    }

    Matrix unscale_grad(const Matrix& grad) const {
        return grad * (1.0 / scale_);
    }

    bool has_overflow(const Matrix& grad) const {
        for (const auto& v : grad.data)
            if (std::isinf(v) || std::isnan(v)) return true;
        return false;
    }

    bool step(const Matrix& grad) {
        total_steps_++;
        if (has_overflow(grad)) {
            scale_ *= backoff_factor_;
            steps_ok_ = 0;
            overflow_count_++;
            if (scale_ < 1.0) scale_ = 1.0;
            return false;
        }
        steps_ok_++;
        if (steps_ok_ >= growth_interval_) {
            scale_ *= growth_factor_;
            steps_ok_ = 0;
            if (scale_ > 1e18) scale_ = 1e18;
        }
        return true;
    }

    void print_status() const {
        std::cout << util::color::DIM
                  << "  [LossScaler] scale=" << std::scientific << std::setprecision(1) << scale_
                  << " overflows=" << overflow_count_
                  << " steps=" << total_steps_
                  << util::color::RESET << "\n";
    }
};

// Eğitim konfigürasyonu
struct TrainConfig {
    int epochs = 1000;
    double lr = 0.01;
    int batch_size = 32;      // Mini-batch boyutu
    Loss loss = Loss::MSE;
    OptimizerType optimizer = OptimizerType::Adam;
    bool verbose = true;      // Eğitim çıktısı göster
    int print_every = 0;      // Her kaç epoch'ta yazdır (0 = otomatik)
    bool plot = true;         // Loss grafiği çiz
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
};

class Network {
public:
    std::vector<DenseLayer> layers;
    Optimizer optimizer;
    std::vector<double> loss_history;

    Network() = default;

    // Hızlı oluşturma: katman boyutları
    // Örnek: Network({2, 8, 4, 1}) → 2→8→4→1
    // Varsayılan: gizli katmanlar ReLU, çıkış katmanı Sigmoid
    Network(std::initializer_list<size_t> topology) {
        auto sizes = std::vector<size_t>(topology);
        if (sizes.size() < 2) {
            throw std::runtime_error("Ağ en az 2 katman boyutuna sahip olmalı (giriş + çıkış)");
        }

        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            Activation act;
            if (i == sizes.size() - 2) {
                // Çıkış katmanı — sigmoid
                act = Activation::Sigmoid;
            } else {
                // Gizli katmanlar — ReLU
                act = Activation::ReLU;
            }
            layers.emplace_back(sizes[i], sizes[i + 1], act);
        }
    }

    // Gelişmiş oluşturma: katman boyutları + aktivasyonlar
    Network(std::vector<size_t> sizes, std::vector<Activation> activations) {
        if (sizes.size() < 2 || activations.size() != sizes.size() - 1) {
            throw std::runtime_error("Boyut ve aktivasyon sayıları uyuşmalı");
        }
        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            layers.emplace_back(sizes[i], sizes[i + 1], activations[i]);
        }
    }

    // Katman ekle
    Network& add(size_t input_size, size_t output_size, Activation act = Activation::ReLU) {
        layers.emplace_back(input_size, output_size, act);
        return *this;
    }

    // ====== İleri Yönlü Hesaplama ======
    Matrix forward(const Matrix& input) {
        Matrix current = input;
        for (auto& layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    // ====== Geri Yayılım ======
    Matrix backward(const Matrix& loss_grad) {
        Matrix grad = loss_grad;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            grad = layers[i].backward(grad);
        }
        return grad;
    }

    // ====== Ağırlık Güncelleme ======
    void update_weights() {
        for (auto& layer : layers) {
            layer.update_weights(optimizer);
        }
    }

    // ====== Tahmin ======
    std::vector<double> predict(const std::vector<double>& input) {
        Matrix inp = Matrix::from_vector(input).T();  // Satır vektörü
        Matrix out = forward(inp);
        return out.to_vector();
    }

    // ====== Eğitim ======
    void train(const Dataset& data, TrainConfig config = {}) {
        // Optimizer kur
        optimizer = Optimizer(OptimizerConfig{
            .type = config.optimizer,
            .lr = config.lr,
            .momentum = config.momentum,
            .beta1 = config.beta1,
            .beta2 = config.beta2
        });

        loss_history.clear();

        int print_every = config.print_every;
        if (print_every <= 0) {
            print_every = std::max(1, config.epochs / 50);
        }

        int batch_size = config.batch_size;
        if (batch_size <= 0) batch_size = 32;

        if (config.verbose) {
            std::cout << util::color::BOLD << util::color::GREEN
                      << "\n  🚀 Eğitim Başlıyor!\n"
                      << util::color::RESET
                      << util::color::DIM
                      << "  ─────────────────────────────────\n"
                      << "  Epochs:    " << config.epochs << "\n"
                      << "  Batch Size:" << batch_size << "\n"
                      << "  LR:        " << config.lr << "\n"
                      << "  Loss:      " << loss::name(config.loss) << "\n"
                      << "  Optimizer: " << optimizer.name() << "\n"
                      << "  Veri:      " << data.size() << " örnek\n"
                      << "  ─────────────────────────────────\n\n"
                      << util::color::RESET;
        }

        util::Timer timer;

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            double epoch_loss = 0.0;

            for (size_t i = 0; i < data.size(); i += batch_size) {
                size_t end_idx = std::min(i + static_cast<size_t>(batch_size), data.size());
                size_t current_batch = end_idx - i;

                Matrix inp(current_batch, data[0].first.size());
                Matrix tgt(current_batch, data[0].second.size());

                for (size_t b = 0; b < current_batch; ++b) {
                    for (size_t j = 0; j < data[i + b].first.size(); ++j)
                        inp(b, j) = data[i + b].first[j];
                    for (size_t j = 0; j < data[i + b].second.size(); ++j)
                        tgt(b, j) = data[i + b].second[j];
                }

                // İleri
                Matrix output = forward(inp);

                // Kayıp hesapla
                epoch_loss += loss::compute(config.loss, output, tgt) * static_cast<double>(current_batch);

                // Geri
                Matrix grad = loss::gradient(config.loss, output, tgt);
                backward(grad);

                // Güncelle
                update_weights();
            }

            epoch_loss /= static_cast<double>(data.size());
            loss_history.push_back(epoch_loss);

            if (config.verbose && (epoch % print_every == 0 || epoch == 1 || epoch == config.epochs)) {
                util::progress_bar(epoch, config.epochs, epoch_loss);
            }
        }

        if (config.verbose) {
            std::cout << "\n\n" << util::color::BOLD << util::color::GREEN
                      << "  ✅ Eğitim Tamamlandı! "
                      << util::color::CYAN << "(" << timer.elapsed_str() << ")\n"
                      << util::color::RESET;

            if (config.plot) {
                util::plot_loss(loss_history);
            }
        }
    }

    // ====== Değerlendirme ======
    double evaluate(const Dataset& data, Loss loss_type = Loss::MSE) {
        double total_loss = 0.0;
        for (const auto& [input, target] : data) {
            Matrix inp = Matrix::row_vector(input);
            Matrix tgt = Matrix::row_vector(target);
            Matrix output = forward(inp);
            total_loss += loss::compute(loss_type, output, tgt);
        }
        return total_loss / static_cast<double>(data.size());
    }

    // ====== Model Özeti ======
    void summary(const std::string& title = "XasmAI Network") const {
        std::vector<std::string> infos;
        size_t total = 0;
        for (const auto& layer : layers) {
            infos.push_back(layer.info());
            total += layer.param_count();
        }
        util::print_summary(title, infos, total);
    }

    // ====== Kaydet ======
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Dosya açılamadı: " + filename);
        }

        // Magic number
        const char magic[] = "XASM";
        file.write(magic, 4);

        // Versiyon
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Katman sayısı
        uint32_t num_layers = static_cast<uint32_t>(layers.size());
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer : layers) {
            // Boyutlar
            uint32_t in_size = static_cast<uint32_t>(layer.input_size());
            uint32_t out_size = static_cast<uint32_t>(layer.output_size());
            uint32_t act = static_cast<uint32_t>(layer.act_type);

            file.write(reinterpret_cast<const char*>(&in_size), sizeof(in_size));
            file.write(reinterpret_cast<const char*>(&out_size), sizeof(out_size));
            file.write(reinterpret_cast<const char*>(&act), sizeof(act));

            // Ağırlıklar
            file.write(reinterpret_cast<const char*>(layer.weights.data.data()),
                       static_cast<std::streamsize>(layer.weights.data.size() * sizeof(double)));

            // Biases
            file.write(reinterpret_cast<const char*>(layer.biases.data.data()),
                       static_cast<std::streamsize>(layer.biases.data.size() * sizeof(double)));
        }

        // Loss history
        uint32_t hist_size = static_cast<uint32_t>(loss_history.size());
        file.write(reinterpret_cast<const char*>(&hist_size), sizeof(hist_size));
        file.write(reinterpret_cast<const char*>(loss_history.data()),
                   static_cast<std::streamsize>(loss_history.size() * sizeof(double)));

        file.close();
        std::cout << util::color::GREEN << "  💾 Model kaydedildi: " << filename
                  << util::color::RESET << "\n";
    }

    // ====== Yükle ======
    static Network load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Dosya açılamadı: " + filename);
        }

        // Magic number doğrula
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "XASM") {
            throw std::runtime_error("Geçersiz dosya formatı!");
        }

        // Versiyon
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        // Katmanları oku
        Network net;
        uint32_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        for (uint32_t i = 0; i < num_layers; ++i) {
            uint32_t in_size, out_size, act;
            file.read(reinterpret_cast<char*>(&in_size), sizeof(in_size));
            file.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));
            file.read(reinterpret_cast<char*>(&act), sizeof(act));

            DenseLayer layer(in_size, out_size, static_cast<Activation>(act));

            // Ağırlıkları oku
            file.read(reinterpret_cast<char*>(layer.weights.data.data()),
                       static_cast<std::streamsize>(layer.weights.data.size() * sizeof(double)));

            file.read(reinterpret_cast<char*>(layer.biases.data.data()),
                       static_cast<std::streamsize>(layer.biases.data.size() * sizeof(double)));

            net.layers.push_back(std::move(layer));
        }

        // Loss history
        uint32_t hist_size;
        file.read(reinterpret_cast<char*>(&hist_size), sizeof(hist_size));
        net.loss_history.resize(hist_size);
        file.read(reinterpret_cast<char*>(net.loss_history.data()),
                   static_cast<std::streamsize>(hist_size * sizeof(double)));

        file.close();
        std::cout << util::color::GREEN << "  📂 Model yüklendi: " << filename
                  << util::color::RESET << "\n";
        return net;
    }

    // ====== Toplam Parametre Sayısı ======
    size_t total_params() const {
        size_t total = 0;
        for (const auto& layer : layers)
            total += layer.param_count();
        return total;
    }
};

} // namespace xasm
