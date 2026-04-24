#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <queue>

namespace xasm {

class BPETokenizer {
    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const {
            return std::hash<int64_t>{}(
                (static_cast<int64_t>(p.first) << 32) | static_cast<int64_t>(p.second));
        }
    };

    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::pair<int,int>> merges;
    size_t vocab_size_;

    std::pair<int,int> most_frequent_pair(const std::vector<std::vector<int>>& corpus) const {
        std::unordered_map<std::pair<int,int>, int, PairHash> counts;
        for (const auto& seq : corpus)
            for (size_t i = 0; i + 1 < seq.size(); ++i)
                counts[{seq[i], seq[i+1]}]++;

        std::pair<int,int> best = {-1, -1};
        int best_count = 0;
        for (const auto& [pair, count] : counts)
            if (count > best_count) { best_count = count; best = pair; }
        return best;
    }

    void apply_merge(std::vector<std::vector<int>>& corpus, std::pair<int,int> pair, int new_id) {
        for (auto& seq : corpus) {
            std::vector<int> merged;
            merged.reserve(seq.size());
            size_t i = 0;
            while (i < seq.size()) {
                if (i + 1 < seq.size() && seq[i] == pair.first && seq[i+1] == pair.second) {
                    merged.push_back(new_id);
                    i += 2;
                } else {
                    merged.push_back(seq[i]);
                    i++;
                }
            }
            seq = std::move(merged);
        }
    }

public:
    BPETokenizer() : vocab_size_(0) {}

    void train(const std::string& text, size_t target_vocab_size) {
        token_to_id.clear();
        id_to_token.clear();
        merges.clear();

        std::set<unsigned char> chars;
        for (unsigned char c : text) chars.insert(c);

        int next_id = 0;
        for (unsigned char c : chars) {
            std::string s(1, static_cast<char>(c));
            token_to_id[s] = next_id;
            id_to_token[next_id] = s;
            next_id++;
        }

        std::vector<std::vector<int>> corpus;
        {
            std::vector<int> seq;
            seq.reserve(text.size());
            for (unsigned char c : text)
                seq.push_back(token_to_id[std::string(1, static_cast<char>(c))]);
            corpus.push_back(std::move(seq));
        }

        while (static_cast<size_t>(next_id) < target_vocab_size) {
            auto best = most_frequent_pair(corpus);
            if (best.first < 0) break;

            std::string merged_str = id_to_token[best.first] + id_to_token[best.second];
            token_to_id[merged_str] = next_id;
            id_to_token[next_id] = merged_str;
            merges.push_back(best);

            apply_merge(corpus, best, next_id);
            next_id++;
        }

        vocab_size_ = static_cast<size_t>(next_id);
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            std::string s(1, static_cast<char>(c));
            auto it = token_to_id.find(s);
            if (it != token_to_id.end())
                ids.push_back(it->second);
        }

        for (const auto& merge : merges) {
            std::vector<int> merged;
            merged.reserve(ids.size());
            size_t i = 0;
            int new_id = token_to_id.at(id_to_token.at(merge.first) + id_to_token.at(merge.second));
            while (i < ids.size()) {
                if (i + 1 < ids.size() && ids[i] == merge.first && ids[i+1] == merge.second) {
                    merged.push_back(new_id);
                    i += 2;
                } else {
                    merged.push_back(ids[i]);
                    i++;
                }
            }
            ids = std::move(merged);
        }

        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            auto it = id_to_token.find(id);
            if (it != id_to_token.end())
                result += it->second;
        }
        return result;
    }

    size_t vocab_size() const { return vocab_size_; }

    bool save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t vs = static_cast<uint32_t>(vocab_size_);
        file.write(reinterpret_cast<const char*>(&vs), 4);

        for (uint32_t i = 0; i < vs; ++i) {
            const std::string& tok = id_to_token.at(static_cast<int>(i));
            uint32_t len = static_cast<uint32_t>(tok.size());
            file.write(reinterpret_cast<const char*>(&len), 4);
            file.write(tok.data(), static_cast<std::streamsize>(len));
        }

        uint32_t nm = static_cast<uint32_t>(merges.size());
        file.write(reinterpret_cast<const char*>(&nm), 4);
        for (const auto& m : merges) {
            int32_t a = m.first, b = m.second;
            file.write(reinterpret_cast<const char*>(&a), 4);
            file.write(reinterpret_cast<const char*>(&b), 4);
        }

        file.close();
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t vs;
        file.read(reinterpret_cast<char*>(&vs), 4);
        vocab_size_ = vs;

        token_to_id.clear();
        id_to_token.clear();

        for (uint32_t i = 0; i < vs; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), 4);
            std::string tok(len, '\0');
            file.read(tok.data(), len);
            token_to_id[tok] = static_cast<int>(i);
            id_to_token[static_cast<int>(i)] = tok;
        }

        uint32_t nm;
        file.read(reinterpret_cast<char*>(&nm), 4);
        merges.resize(nm);
        for (uint32_t i = 0; i < nm; ++i) {
            int32_t a, b;
            file.read(reinterpret_cast<char*>(&a), 4);
            file.read(reinterpret_cast<char*>(&b), 4);
            merges[i] = {a, b};
        }

        file.close();
        return true;
    }
};

} // namespace xasm
