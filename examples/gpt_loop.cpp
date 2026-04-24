#include "../include/xasmai/transformer.hpp"
#include "../include/xasmai/dataloader.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <set>
#include <csignal>
#include <filesystem>

static volatile sig_atomic_t g_shutdown = 0;
static std::string g_checkpoint_path;

void signal_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
}

struct Tokenizer {
    std::unordered_map<char, int> char_to_id;
    std::unordered_map<int, char> id_to_char;
    size_t vocab_size = 0;

    void build(const std::string& text) {
        std::set<char> chars(text.begin(), text.end());
        int id = 0;
        for (char c : chars) {
            char_to_id[c] = id;
            id_to_char[id] = c;
            id++;
        }
        vocab_size = static_cast<size_t>(id);
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = char_to_id.find(c);
            if (it != char_to_id.end())
                ids.push_back(it->second);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        text.reserve(ids.size());
        for (int id : ids) {
            auto it = id_to_char.find(id);
            if (it != id_to_char.end())
                text += it->second;
        }
        return text;
    }

    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        uint32_t vs = static_cast<uint32_t>(vocab_size);
        f.write(reinterpret_cast<const char*>(&vs), 4);
        for (const auto& [ch, id] : char_to_id) {
            f.write(&ch, 1);
            int32_t i = id;
            f.write(reinterpret_cast<const char*>(&i), 4);
        }
        f.close();
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        uint32_t vs;
        f.read(reinterpret_cast<char*>(&vs), 4);
        vocab_size = vs;
        char_to_id.clear();
        id_to_char.clear();
        for (uint32_t i = 0; i < vs; ++i) {
            char ch;
            int32_t id;
            f.read(&ch, 1);
            f.read(reinterpret_cast<char*>(&id), 4);
            char_to_id[ch] = id;
            id_to_char[id] = ch;
        }
        f.close();
        return true;
    }
};

class CSVLogger {
    std::ofstream file;
    bool header_written = false;

public:
    CSVLogger() = default;

    bool open(const std::string& path, bool append = true) {
        if (append && std::filesystem::exists(path)) {
            file.open(path, std::ios::app);
            header_written = true;
        } else {
            file.open(path, std::ios::out);
        }
        return file.is_open();
    }

    void write_header() {
        if (!header_written) {
            file << "step,loss,loss_ema,ppl,tok_per_sec,lr,elapsed_sec,total_tokens,epoch\n";
            header_written = true;
            file.flush();
        }
    }

    void log(size_t step, double loss, double loss_ema, double ppl,
             double tps, double lr, double elapsed, size_t total_tokens, size_t epoch) {
        file << step << ","
             << std::fixed << std::setprecision(6) << loss << ","
             << loss_ema << ","
             << std::setprecision(2) << ppl << ","
             << std::setprecision(0) << tps << ","
             << std::scientific << std::setprecision(4) << lr << ","
             << std::fixed << std::setprecision(1) << elapsed << ","
             << total_tokens << ","
             << epoch << "\n";
        file.flush();
    }

    void log_generation(size_t step, const std::string& text) {
        std::string safe = text;
        for (auto& c : safe) if (c == '\n' || c == '\r' || c == ',') c = ' ';
        file << "# GEN step=" << step << ": " << safe << "\n";
        file.flush();
    }
};

void print_banner() {
    std::cout << "\033[2J\033[H";
    std::cout << xasm::util::color::BOLD << xasm::util::color::CYAN << R"(
     ██████╗ ██████╗ ████████╗
    ██╔════╝ ██╔══██╗╚══██╔══╝
    ██║  ███╗██████╔╝   ██║
    ██║   ██║██╔═══╝    ██║
    ╚██████╔╝██║        ██║
     ╚═════╝ ╚═╝        ╚═╝
)" << xasm::util::color::RESET;
}

void emergency_save(xasm::GPTModel& model, const xasm::TrainState& ts) {
    std::cout << "\n" << xasm::util::color::YELLOW
              << "  ⚡ Sinyal alindi. Checkpoint kaydediliyor..."
              << xasm::util::color::RESET << "\n";

    std::string path = g_checkpoint_path.empty() ? "checkpoint_emergency.bin" : g_checkpoint_path;
    if (model.save_checkpoint(path, ts))
        std::cout << xasm::util::color::GREEN
                  << "  ✅ Checkpoint kaydedildi: " << path
                  << xasm::util::color::RESET << "\n";
    else
        std::cout << xasm::util::color::RED
                  << "  ❌ Checkpoint kaydi basarisiz!"
                  << xasm::util::color::RESET << "\n";
}

void rotating_save(xasm::GPTModel& model, const xasm::TrainState& ts,
                    const std::string& base_path) {
    std::string prev = base_path + "_prev.bin";
    std::string latest = base_path + "_latest.bin";

    if (std::filesystem::exists(latest))
        std::filesystem::rename(latest, prev);

    model.save_checkpoint(latest, ts);
}

int main(int argc, char* argv[]) {
    using namespace xasm;

    std::string data_path = "data.txt";
    std::string resume_path = "";
    size_t SEQ_LEN   = 64;
    size_t D_MODEL   = 64;
    size_t N_HEADS   = 4;
    size_t N_KV_HEADS = 4;
    size_t N_LAYERS  = 2;
    size_t D_FF      = 256;
    double LR        = 3e-4;
    double MIN_LR    = 1e-5;
    double WD        = 0.01;
    size_t WARMUP    = 1000;
    size_t TOTAL_STEPS = 10000000;
    size_t GRAD_ACCUM = 1;
    size_t LOG_EVERY = 10;
    size_t GEN_EVERY = 500;
    size_t SAVE_EVERY = 5000;
    size_t GEN_LEN   = 100;
    int TOP_K        = 40;
    double TOP_P     = 0.9;
    double REP_PEN   = 1.1;
    size_t N_EXPERTS = 1;
    size_t ACTIVE_EXP = 1;
    double BAL_COEFF = 0.01;
    size_t WINDOW    = 0;
    bool GRAD_CKPT   = false;
    std::string CKPT_BASE = "checkpoint";
    std::string LOG_FILE = "train_log.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--seq" && i + 1 < argc) SEQ_LEN = std::stoull(argv[++i]);
        else if (arg == "--dim" && i + 1 < argc) D_MODEL = std::stoull(argv[++i]);
        else if (arg == "--heads" && i + 1 < argc) N_HEADS = std::stoull(argv[++i]);
        else if (arg == "--kv-heads" && i + 1 < argc) N_KV_HEADS = std::stoull(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) N_LAYERS = std::stoull(argv[++i]);
        else if (arg == "--ff" && i + 1 < argc) D_FF = std::stoull(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) LR = std::stod(argv[++i]);
        else if (arg == "--min-lr" && i + 1 < argc) MIN_LR = std::stod(argv[++i]);
        else if (arg == "--wd" && i + 1 < argc) WD = std::stod(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) WARMUP = std::stoull(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) TOTAL_STEPS = std::stoull(argv[++i]);
        else if (arg == "--accum" && i + 1 < argc) GRAD_ACCUM = std::stoull(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) TOP_K = std::stoi(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) TOP_P = std::stod(argv[++i]);
        else if (arg == "--rep-penalty" && i + 1 < argc) REP_PEN = std::stod(argv[++i]);
        else if (arg == "--experts" && i + 1 < argc) N_EXPERTS = std::stoull(argv[++i]);
        else if (arg == "--active-experts" && i + 1 < argc) ACTIVE_EXP = std::stoull(argv[++i]);
        else if (arg == "--balance-coeff" && i + 1 < argc) BAL_COEFF = std::stod(argv[++i]);
        else if (arg == "--window" && i + 1 < argc) WINDOW = std::stoull(argv[++i]);
        else if (arg == "--grad-ckpt") GRAD_CKPT = true;
        else if (arg == "--log-every" && i + 1 < argc) LOG_EVERY = std::stoull(argv[++i]);
        else if (arg == "--gen-every" && i + 1 < argc) GEN_EVERY = std::stoull(argv[++i]);
        else if (arg == "--save-every" && i + 1 < argc) SAVE_EVERY = std::stoull(argv[++i]);
        else if (arg == "--ckpt" && i + 1 < argc) CKPT_BASE = argv[++i];
        else if (arg == "--log" && i + 1 < argc) LOG_FILE = argv[++i];
        else if (arg[0] != '-') data_path = arg;
    }

    g_checkpoint_path = CKPT_BASE + "_latest.bin";

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::ifstream file(data_path);
    if (!file.is_open()) {
        std::cout << util::color::RED
                  << "\n  Kaynak dosya bulunamadi: " << data_path << "\n"
                  << "  Kullanim: ./gpt_loop <metin_dosyasi> [seçenekler]\n"
                  << "  Seçenekler:\n"
                  << "    --resume <path>    Checkpoint'tan devam et\n"
                  << "    --seq <n>          Sekans uzunlugu (varsayilan: 64)\n"
                  << "    --dim <n>          Model boyutu (varsayilan: 64)\n"
                  << "    --heads <n>        Dikkat basi sayisi (varsayilan: 4)\n"
                  << "    --layers <n>       Katman sayisi (varsayilan: 2)\n"
                  << "    --lr <f>           Ogrenme orani (varsayilan: 3e-4)\n"
                  << "    --warmup <n>       Warmup adimlari (varsayilan: 1000)\n"
                  << "    --accum <n>        Gradyan birikimi (varsayilan: 1)\n"
                  << "    --save-every <n>   Checkpoint araligi (varsayilan: 5000)\n"
                  << util::color::RESET;
        return 1;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string corpus = ss.str();
    file.close();

    if (corpus.size() < 100) {
        std::cout << util::color::RED << "\n  Kaynak cok kisa. En az 100 karakter gerekli.\n\n"
                  << util::color::RESET;
        return 1;
    }

    Tokenizer tok;
    tok.build(corpus);
    std::vector<int> all_ids = tok.encode(corpus);

    MMapDataLoader loader(SEQ_LEN, 42);
    loader.set_token_ids(all_ids);

    TrainState ts;
    GPTModel model;

    if (!resume_path.empty() && std::filesystem::exists(resume_path)) {
        print_banner();
        std::cout << util::color::YELLOW
                  << "    Checkpoint yukleniyor: " << resume_path << "\n"
                  << util::color::RESET;

        if (model.load_checkpoint(resume_path, ts)) {
            std::cout << util::color::GREEN
                      << "    ✅ Checkpoint yuklendi: step=" << ts.step
                      << " loss_ema=" << std::fixed << std::setprecision(4) << ts.loss_ema << "\n"
                      << util::color::RESET;
        } else {
            std::cout << util::color::RED << "    ❌ Checkpoint yuklenemedi!\n" << util::color::RESET;
            return 1;
        }
    } else {
        GPTConfig cfg{
            .vocab_size = tok.vocab_size,
            .d_model = D_MODEL,
            .num_heads = N_HEADS,
            .num_kv_heads = N_KV_HEADS,
            .num_layers = N_LAYERS,
            .d_ff = D_FF,
            .max_seq_len = SEQ_LEN,
            .lr = LR,
            .grad_clip = 1.0,
            .weight_decay = WD,
            .min_lr = MIN_LR,
            .warmup_steps = WARMUP,
            .total_steps = TOTAL_STEPS,
            .grad_accum_steps = GRAD_ACCUM,
            .top_k = TOP_K,
            .top_p = TOP_P,
            .repetition_penalty = REP_PEN,
            .num_experts = N_EXPERTS,
            .active_experts = ACTIVE_EXP,
            .moe_balance_coeff = BAL_COEFF,
            .window_size = WINDOW,
            .grad_checkpoint = static_cast<uint8_t>(GRAD_CKPT ? 1 : 0)
        };
        model = GPTModel(cfg);
        print_banner();
    }

    std::cout << util::color::DIM
              << "    XasmAI Transformer — Long-Run Training Engine\n"
              << "    Kaynak: " << data_path << " (" << corpus.size() << " byte, "
              << all_ids.size() << " token)\n"
              << "    Vocab: " << tok.vocab_size
              << " | Seq: " << SEQ_LEN
              << " | GradAccum: " << GRAD_ACCUM << "\n"
              << util::color::RESET;

    model.summary();

    CSVLogger csv;
    csv.open(LOG_FILE, !resume_path.empty());
    csv.write_header();

    std::cout << util::color::BOLD
              << "  ┌──────────┬────────────┬──────────┬──────────┬───────────┬────────┐\n"
              << "  │   Step   │    Loss    │  PPL     │  tok/s   │   LR      │ Epoch  │\n"
              << "  ├──────────┼────────────┼──────────┼──────────┼───────────┼────────┤\n"
              << util::color::RESET;

    std::mt19937 rng(42);
    auto wall_start = std::chrono::high_resolution_clock::now();

    for (size_t step = ts.step + 1; !g_shutdown; ++step) {
        model.zero_grad();
        double accum_loss = 0.0;

        for (size_t a = 0; a < GRAD_ACCUM; ++a) {
            auto batch = loader.next_batch();
            double loss = model.compute_loss_backward(batch.input, batch.target);
            accum_loss += loss;
        }
        accum_loss /= static_cast<double>(GRAD_ACCUM);

        model.apply_update(GRAD_ACCUM);

        ts.step = step;
        ts.total_tokens += SEQ_LEN * GRAD_ACCUM;

        if (ts.loss_ema == 0.0) ts.loss_ema = accum_loss;
        else ts.loss_ema = 0.95 * ts.loss_ema + 0.05 * accum_loss;

        if (step % LOG_EVERY == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            double tps = static_cast<double>(ts.total_tokens) / elapsed;
            double ppl = std::exp(std::min(ts.loss_ema, 20.0));
            double lr = model.current_lr();

            std::string loss_color;
            if (ts.loss_ema < 2.0) loss_color = util::color::GREEN;
            else if (ts.loss_ema < 4.0) loss_color = util::color::YELLOW;
            else loss_color = util::color::RED;

            std::cout << "  │ " << std::setw(8) << step << " │"
                      << loss_color << std::setw(11) << std::fixed << std::setprecision(4) << ts.loss_ema
                      << util::color::RESET << " │"
                      << std::setw(9) << std::setprecision(1) << ppl << " │"
                      << util::color::CYAN << std::setw(9) << std::setprecision(0) << tps
                      << util::color::RESET << " │"
                      << util::color::DIM << std::setw(10) << std::scientific << std::setprecision(1) << lr
                      << util::color::RESET << " │"
                      << std::setw(6) << loader.epoch()
                      << " │\n";

            csv.log(step, accum_loss, ts.loss_ema, ppl, tps, lr, elapsed, ts.total_tokens, loader.epoch());
        }

        if (step % GEN_EVERY == 0) {
            std::cout << util::color::DIM
                      << "  ├──────────┴────────────┴──────────┴──────────┴───────────┴────────┤\n"
                      << util::color::RESET;

            std::vector<int> prompt_ids;
            std::uniform_int_distribution<size_t> p_dist(0, all_ids.size() - 10);
            size_t p_start = p_dist(rng);
            for (size_t i = 0; i < 8 && p_start + i < all_ids.size(); ++i)
                prompt_ids.push_back(all_ids[p_start + i]);

            auto generated = model.generate(prompt_ids, GEN_LEN, 0.8);
            std::string gen_text = tok.decode(generated);

            if (gen_text.size() > 68) gen_text = gen_text.substr(0, 68);
            for (auto& c : gen_text) if (c == '\n' || c == '\r') c = ' ';

            std::cout << util::color::MAGENTA << util::color::BOLD
                      << "  │ GEN: " << util::color::RESET
                      << util::color::WHITE << gen_text
                      << std::string(std::max(0, 61 - static_cast<int>(gen_text.size())), ' ')
                      << util::color::RESET << " │\n";

            csv.log_generation(step, gen_text);

            std::cout << util::color::DIM
                      << "  ├──────────┬────────────┬──────────┬──────────┬───────────┬────────┤\n"
                      << util::color::RESET;
        }

        if (step % SAVE_EVERY == 0) {
            rotating_save(model, ts, CKPT_BASE);
            std::cout << util::color::DIM
                      << "  │ 💾 Checkpoint kaydedildi: " << CKPT_BASE << "_latest.bin"
                      << std::string(32, ' ') << "│\n"
                      << util::color::RESET;
        }
    }

    emergency_save(model, ts);
    tok.save("tokenizer.bin");

    std::cout << "\n" << util::color::BOLD << util::color::GREEN
              << "  Egitim tamamlandi. Step: " << ts.step
              << " | Loss: " << std::fixed << std::setprecision(4) << ts.loss_ema
              << "\n" << util::color::RESET;

    return 0;
}
