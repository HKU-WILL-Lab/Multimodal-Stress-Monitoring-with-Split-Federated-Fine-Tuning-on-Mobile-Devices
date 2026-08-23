#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace sflclean {

struct StepMetrics {
    std::string run_id;
    std::string client_id;
    std::uint64_t global_round = 0;
    std::uint64_t local_step = 0;
    std::uint32_t batch_size = 0;
    std::uint32_t sequence_length = 0;
    float loss = 0.0F;
    float token_accuracy = 0.0F;
    std::uint64_t split_rpc_bytes = 0;
    std::uint64_t duration_ms = 0;
};

class JsonlMetricWriter {
public:
    explicit JsonlMetricWriter(const std::string& path);
    void append(const StepMetrics& metrics);

private:
    std::ofstream output_;
};

}  // namespace sflclean
