#include "sfl/wiki_text_stream.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "sfl_wikitext_stream_test.txt";
    {
        std::ofstream output(path, std::ios::binary);
        output << "zero\n" << "one\n" << "two\n" << "three\n";
    }
    auto encode = [](const std::string& text) {
        return std::vector<int>{static_cast<int>(text.size())};
    };
    sflclean::WikiTextStream stream(path.string(), 1, 2, 9, 0, encode);
    auto first = stream.next_batch(1, 3);
    if (first.batch_size != 1 || first.token_ids != std::vector<std::int64_t>({3, 9, 5}) ||
        first.attention_mask != std::vector<std::int64_t>({1, 1, 1})) {
        throw std::runtime_error("round-robin assignment or carry handling failed");
    }
    auto final = stream.next_batch(1, 3);
    if (final.batch_size != 1 || final.token_ids != std::vector<std::int64_t>({9, 0, 0}) ||
        final.attention_mask != std::vector<std::int64_t>({1, 0, 0})) {
        throw std::runtime_error("final-only padding failed");
    }
    if (stream.next_batch(1, 3).batch_size != 0) {
        throw std::runtime_error("stream emitted data after final sequence");
    }
    std::filesystem::remove(path);
    std::cout << "WikiTextStream tests passed\n";
    return 0;
}
