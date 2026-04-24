#pragma once

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace xasm {

class MMapDataLoader {
    struct MappedFile {
        const char* data = nullptr;
        size_t size = 0;
        int fd = -1;
    };

    std::vector<MappedFile> files;
    std::vector<int> all_ids;
    size_t total_chars = 0;
    size_t seq_len;
    std::mt19937 rng;
    size_t epoch_ = 0;
    size_t samples_in_epoch = 0;
    size_t samples_per_epoch = 0;

    void map_file(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return;

        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); return; }

        size_t sz = static_cast<size_t>(st.st_size);
        if (sz == 0) { close(fd); return; }

        void* ptr = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); return; }

        madvise(ptr, sz, MADV_SEQUENTIAL);

        files.push_back({static_cast<const char*>(ptr), sz, fd});
        total_chars += sz;
    }

public:
    MMapDataLoader() : seq_len(64), rng(42) {}

    MMapDataLoader(size_t sl, uint32_t seed = 42) : seq_len(sl), rng(seed) {}

    ~MMapDataLoader() {
        for (auto& f : files) {
            if (f.data) munmap(const_cast<char*>(f.data), f.size);
            if (f.fd >= 0) close(f.fd);
        }
    }

    MMapDataLoader(const MMapDataLoader&) = delete;
    MMapDataLoader& operator=(const MMapDataLoader&) = delete;

    bool load_file(const std::string& path) {
        map_file(path);
        return !files.empty();
    }

    size_t load_directory(const std::string& dir_path) {
        size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".txt" || ext == ".md" || ext == ".csv" || ext == ".json") {
                    map_file(entry.path().string());
                    count++;
                }
            }
        }
        return count;
    }

    void preprocess(const std::function<std::vector<int>(const std::string&)>& encode_fn) {
        for (const auto& f : files) {
            std::string text(f.data, f.size);
            auto ids = encode_fn(text);
            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
        if (all_ids.size() > seq_len + 1)
            samples_per_epoch = all_ids.size() - seq_len - 1;
    }

    void set_token_ids(std::vector<int> ids) {
        all_ids = std::move(ids);
        if (all_ids.size() > seq_len + 1)
            samples_per_epoch = all_ids.size() - seq_len - 1;
    }

    struct Batch {
        std::vector<int> input;
        std::vector<int> target;
    };

    Batch next_batch() {
        if (all_ids.size() <= seq_len + 1)
            return {{}, {}};

        size_t max_start = all_ids.size() - seq_len - 1;
        std::uniform_int_distribution<size_t> dist(0, max_start);
        size_t start = dist(rng);

        Batch b;
        b.input.assign(all_ids.begin() + static_cast<long>(start),
                        all_ids.begin() + static_cast<long>(start + seq_len));
        b.target.assign(all_ids.begin() + static_cast<long>(start + 1),
                         all_ids.begin() + static_cast<long>(start + seq_len + 1));

        samples_in_epoch++;
        if (samples_in_epoch >= samples_per_epoch) {
            epoch_++;
            samples_in_epoch = 0;
        }

        return b;
    }

    size_t total_tokens() const { return all_ids.size(); }
    size_t total_bytes() const { return total_chars; }
    size_t num_files() const { return files.size(); }
    size_t epoch() const { return epoch_; }

    std::string raw_text(size_t offset, size_t len) const {
        std::string result;
        size_t pos = 0;
        for (const auto& f : files) {
            if (offset < pos + f.size) {
                size_t start = offset - pos;
                size_t chunk = std::min(len, f.size - start);
                result.append(f.data + start, chunk);
                len -= chunk;
                offset += chunk;
                if (len == 0) break;
            }
            pos += f.size;
        }
        return result;
    }
};

} // namespace xasm
