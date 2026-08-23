#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace sflclean {

struct TokenBatch {
    std::uint32_t batch_size = 0;
    std::uint32_t sequence_length = 0;
    std::uint64_t processed_sequences = 0;
    std::vector<std::int64_t> token_ids;
    std::vector<std::int64_t> attention_mask;
};

class WikiTextStream {
public:
    using EncodeFunction = std::function<std::vector<int>(const std::string&)>;

    WikiTextStream(const std::string& path,
                   std::uint32_t client_index,
                   std::uint32_t client_count,
                   int eos_token,
                   int pad_token,
                   EncodeFunction encode);

    TokenBatch next_batch(std::uint32_t batch_size,
                          std::uint32_t sequence_length);

    std::uint64_t source_lines_read() const { return source_line_index_; }

private:
    bool fill_one_sequence(std::uint32_t sequence_length,
                           std::vector<std::int64_t>& ids,
                           std::vector<std::int64_t>& mask);
    bool read_assigned_line();

    std::ifstream input_;
    std::uint32_t client_index_;
    std::uint32_t client_count_;
    int eos_token_;
    int pad_token_;
    EncodeFunction encode_;
    std::deque<std::int64_t> carry_;
    std::uint64_t source_line_index_ = 0;
    bool exhausted_ = false;
    bool final_sequence_emitted_ = false;
};

}  // namespace sflclean
