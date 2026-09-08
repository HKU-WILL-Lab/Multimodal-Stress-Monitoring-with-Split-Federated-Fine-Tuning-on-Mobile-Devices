#include "sfl/sensor_text_stream.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace sflclean {
namespace {

const google::protobuf::ListValue& array_field(
    const google::protobuf::Struct& object, const char* key) {
    const auto found = object.fields().find(key);
    if (found == object.fields().end() ||
        found->second.kind_case() != google::protobuf::Value::kListValue) {
        throw std::invalid_argument(std::string("sensor dataset field must be an array: ") + key);
    }
    return found->second.list_value();
}

std::vector<float> float_array(const google::protobuf::Struct& object, const char* key) {
    std::vector<float> output;
    for (const auto& value : array_field(object, key).values()) {
        if (value.kind_case() != google::protobuf::Value::kNumberValue ||
            !std::isfinite(value.number_value())) {
            throw std::invalid_argument(std::string("dataset contains non-finite numeric value in ") + key);
        }
        output.push_back(static_cast<float>(value.number_value()));
    }
    return output;
}

std::vector<std::int64_t> integer_array(
    const google::protobuf::Struct& object, const char* key, bool binary) {
    std::vector<std::int64_t> output;
    for (const auto& value : array_field(object, key).values()) {
        if (value.kind_case() != google::protobuf::Value::kNumberValue ||
            !std::isfinite(value.number_value()) || std::floor(value.number_value()) != value.number_value() ||
            value.number_value() < 0.0 ||
            value.number_value() > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(std::string("dataset contains invalid integer in ") + key);
        }
        const auto item = static_cast<std::int64_t>(value.number_value());
        if (binary && item != 0 && item != 1) {
            throw std::invalid_argument(std::string("dataset mask is not binary: ") + key);
        }
        output.push_back(item);
    }
    return output;
}

}  // namespace

SensorTextStream::SensorTextStream(std::string path,
                                   std::uint32_t client_index,
                                   std::uint32_t client_count,
                                   std::uint32_t sensor_values_per_example,
                                   std::uint32_t sequence_length)
    : sensor_values_per_example_(sensor_values_per_example),
      sequence_length_(sequence_length) {
    if (client_count == 0 || client_index >= client_count ||
        sensor_values_per_example == 0 || sequence_length < 2) {
        throw std::invalid_argument("invalid sensor dataset partition or shape");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open labeled sensor dataset: " + path);
    std::string line;
    std::uint64_t source_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        google::protobuf::Struct object;
        const auto status = google::protobuf::util::JsonStringToMessage(line, &object);
        if (!status.ok()) throw std::invalid_argument("invalid sensor dataset JSONL record");
        if (source_index++ % client_count != client_index) continue;
        Example example;
        example.sensor = float_array(object, "sensor");
        example.token_ids = integer_array(object, "token_ids", false);
        example.attention_mask = integer_array(object, "attention_mask", true);
        example.loss_mask = integer_array(object, "loss_mask", true);
        if (example.sensor.size() != sensor_values_per_example_ ||
            example.token_ids.size() != sequence_length_ ||
            example.attention_mask.size() != sequence_length_ ||
            example.loss_mask.size() != sequence_length_) {
            throw std::invalid_argument("sensor dataset record does not match configured shapes");
        }
        for (std::size_t i = 0; i < sequence_length_; ++i) {
            if (example.loss_mask[i] > example.attention_mask[i]) {
                throw std::invalid_argument("loss_mask cannot include a hidden text token");
            }
        }
        examples_.push_back(std::move(example));
    }
    if (examples_.empty()) throw std::invalid_argument("client sensor partition is empty");
}

SensorTextBatch SensorTextStream::next_batch(std::uint32_t requested_batch) {
    if (requested_batch == 0) throw std::invalid_argument("requested batch must be positive");
    SensorTextBatch result;
    result.batch_size = requested_batch;
    result.sequence_length = sequence_length_;
    result.sensor_values_per_example = sensor_values_per_example_;
    result.processed_sequences = requested_batch;
    for (std::uint32_t i = 0; i < requested_batch; ++i) {
        const auto& example = examples_[cursor_++ % examples_.size()];
        result.sensor_values.insert(result.sensor_values.end(), example.sensor.begin(), example.sensor.end());
        result.token_ids.insert(result.token_ids.end(), example.token_ids.begin(), example.token_ids.end());
        result.attention_mask.insert(result.attention_mask.end(), example.attention_mask.begin(), example.attention_mask.end());
        result.loss_mask.insert(result.loss_mask.end(), example.loss_mask.begin(), example.loss_mask.end());
    }
    return result;
}

}  // namespace sflclean
