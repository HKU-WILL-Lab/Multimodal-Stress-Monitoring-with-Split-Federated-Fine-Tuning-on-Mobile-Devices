#include "sfl/wiki_text_stream.h"

#include <stdexcept>
#include <utility>

namespace sflclean {

WikiTextStream::WikiTextStream(const std::string& path,
                               std::uint32_t client_index,
                               std::uint32_t client_count,
                               int eos_token,
                               int pad_token,
                               EncodeFunction encode)
    : input_(path, std::ios::binary),
      client_index_(client_index),
      client_count_(client_count),
      eos_token_(eos_token),
      pad_token_(pad_token >= 0 ? pad_token : eos_token),
      encode_(std::move(encode)) {
    if (!input_) throw std::runtime_error("cannot open UTF-8 dataset: " + path);
    if (client_count_ == 0 || client_index_ >= client_count_) {
        throw std::invalid_argument("invalid deterministic client partition");
    }
    if (eos_token_ < 0 || pad_token_ < 0 || !encode_) {
        throw std::invalid_argument("stream requires tokenizer EOS/PAD IDs and encode function");
    }
}

bool WikiTextStream::read_assigned_line() {
    std::string line;
    while (std::getline(input_, line)) {
        const std::uint64_t index = source_line_index_++;
        if (index % client_count_ != client_index_ || line.empty()) continue;
        auto tokens = encode_(line);
        for (const int token : tokens) {
            if (token < 0) throw std::runtime_error("tokenizer produced a negative token ID");
            carry_.push_back(token);
        }
        carry_.push_back(eos_token_);
        return true;
    }
    exhausted_ = true;
    return false;
}

bool WikiTextStream::fill_one_sequence(std::uint32_t sequence_length,
                                       std::vector<std::int64_t>& ids,
                                       std::vector<std::int64_t>& mask) {
    while (carry_.size() < sequence_length && !exhausted_) {
        read_assigned_line();
    }
    if (carry_.empty()) return false;

    const auto real_tokens = std::min<std::size_t>(carry_.size(), sequence_length);
    for (std::size_t i = 0; i < real_tokens; ++i) {
        ids.push_back(carry_.front());
        carry_.pop_front();
        mask.push_back(1);
    }
    if (real_tokens < sequence_length) {
        if (!exhausted_ || final_sequence_emitted_) {
            throw std::logic_error("short sequences may only be emitted once at EOF");
        }
        final_sequence_emitted_ = true;
        ids.resize(ids.size() + sequence_length - real_tokens, pad_token_);
        mask.resize(mask.size() + sequence_length - real_tokens, 0);
    }
    return true;
}

TokenBatch WikiTextStream::next_batch(std::uint32_t batch_size,
                                      std::uint32_t sequence_length) {
    if (batch_size == 0 || sequence_length < 2) {
        throw std::invalid_argument("batch size must be positive and sequence length at least two");
    }
    TokenBatch batch;
    batch.sequence_length = sequence_length;
    batch.token_ids.reserve(static_cast<std::size_t>(batch_size) * sequence_length);
    batch.attention_mask.reserve(static_cast<std::size_t>(batch_size) * sequence_length);
    for (std::uint32_t item = 0; item < batch_size; ++item) {
        if (!fill_one_sequence(sequence_length, batch.token_ids, batch.attention_mask)) break;
        ++batch.batch_size;
        ++batch.processed_sequences;
    }
    return batch;
}

}  // namespace sflclean
