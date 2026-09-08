#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sflclean {

struct SensorTextBatch {
    std::uint32_t batch_size = 0;
    std::uint32_t sequence_length = 0;
    std::uint32_t sensor_values_per_example = 0;
    std::uint64_t processed_sequences = 0;
    std::vector<float> sensor_values;
    std::vector<std::int64_t> token_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> loss_mask;
};

// JSONL interchange format used between the server-side dataset exporter and
// native clients. Each line has sensor, token_ids, attention_mask and loss_mask
// arrays. Tokenization therefore stays identical to the selected Llama model.
class SensorTextStream {
public:
    SensorTextStream(std::string path,
                     std::uint32_t client_index,
                     std::uint32_t client_count,
                     std::uint32_t sensor_values_per_example,
                     std::uint32_t sequence_length);

    SensorTextBatch next_batch(std::uint32_t requested_batch);

private:
    struct Example {
        std::vector<float> sensor;
        std::vector<std::int64_t> token_ids;
        std::vector<std::int64_t> attention_mask;
        std::vector<std::int64_t> loss_mask;
    };
    std::vector<Example> examples_;
    std::size_t cursor_ = 0;
    std::uint32_t sensor_values_per_example_;
    std::uint32_t sequence_length_;
};

}  // namespace sflclean
