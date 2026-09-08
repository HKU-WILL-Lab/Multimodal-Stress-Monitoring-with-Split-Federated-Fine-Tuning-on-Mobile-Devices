#include "sfl/metrics.h"

#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace sflclean {
namespace {

std::string json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size());
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20U) throw std::invalid_argument("metric identifiers contain control characters");
                output.push_back(static_cast<char>(c));
        }
    }
    return output;
}

}  // namespace

JsonlMetricWriter::JsonlMetricWriter(const std::string& path)
    : output_(path, std::ios::app) {
    if (!output_) throw std::runtime_error("cannot open append-only metrics file: " + path);
}

void JsonlMetricWriter::append(const StepMetrics& metrics) {
    if (!std::isfinite(metrics.loss) || !std::isfinite(metrics.token_accuracy)) {
        throw std::invalid_argument("metric loss and accuracy must be finite");
    }
    output_ << std::setprecision(9)
            << "{\"schema\":\"sfl.clean.metrics.v1\""
            << ",\"run_id\":\"" << json_escape(metrics.run_id) << "\""
            << ",\"client_id\":\"" << json_escape(metrics.client_id) << "\""
            << ",\"global_round\":" << metrics.global_round
            << ",\"local_step\":" << metrics.local_step
            << ",\"batch_size\":" << metrics.batch_size
            << ",\"sequence_length\":" << metrics.sequence_length
            << ",\"loss\":" << metrics.loss
            << ",\"token_accuracy\":" << metrics.token_accuracy
            << ",\"split_rpc_bytes\":" << metrics.split_rpc_bytes
            << ",\"duration_ms\":" << metrics.duration_ms
            << ",\"host_gpu_memory_bytes\":null}\n";
    output_.flush();
    if (!output_) throw std::runtime_error("failed to append metrics record");
}

}  // namespace sflclean
