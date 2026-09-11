#include "sfl/client_options.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace sflclean {
namespace {

std::uint32_t parse_u32(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid unsigned integer for --" + key + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parse_float(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid finite float for --" + key + ": " + value);
    }
    return parsed;
}

const google::protobuf::Value& required_field(const google::protobuf::Struct& object,
                                               const char* key) {
    const auto found = object.fields().find(key);
    if (found == object.fields().end()) {
        throw std::invalid_argument(std::string("shared config is missing ") + key);
    }
    return found->second;
}

const google::protobuf::Struct& object_field(const google::protobuf::Struct& object,
                                              const char* key) {
    const auto& value = required_field(object, key);
    if (value.kind_case() != google::protobuf::Value::kStructValue) {
        throw std::invalid_argument(std::string("shared config field must be an object: ") + key);
    }
    return value.struct_value();
}

std::string string_field(const google::protobuf::Struct& object, const char* key) {
    const auto& value = required_field(object, key);
    if (value.kind_case() != google::protobuf::Value::kStringValue || value.string_value().empty()) {
        throw std::invalid_argument(std::string("shared config field must be a non-empty string: ") + key);
    }
    return value.string_value();
}

double number_field(const google::protobuf::Struct& object, const char* key) {
    const auto& value = required_field(object, key);
    if (value.kind_case() != google::protobuf::Value::kNumberValue ||
        !std::isfinite(value.number_value())) {
        throw std::invalid_argument(std::string("shared config field must be finite numeric: ") + key);
    }
    return value.number_value();
}

std::uint32_t integer_field(const google::protobuf::Struct& object, const char* key) {
    const double value = number_field(object, key);
    if (value < 0.0 || value > std::numeric_limits<std::uint32_t>::max() ||
        std::floor(value) != value) {
        throw std::invalid_argument(std::string("shared config field must be an unsigned integer: ") + key);
    }
    return static_cast<std::uint32_t>(value);
}

std::string loopback_target(const std::string& bind, const char* key) {
    const auto colon = bind.rfind(':');
    if (colon == std::string::npos || colon + 1 == bind.size()) {
        throw std::invalid_argument(std::string("shared config has invalid bind for ") + key);
    }
    const auto port = parse_u32(bind.substr(colon + 1), key);
    if (port == 0 || port > 65535) {
        throw std::invalid_argument(std::string("shared config port is invalid for ") + key);
    }
    return "127.0.0.1:" + std::to_string(port);
}

void apply_shared_config(const std::string& path, ClientOptions& options) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open shared config: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    google::protobuf::Struct root;
    const auto status = google::protobuf::util::JsonStringToMessage(buffer.str(), &root);
    if (!status.ok()) {
        throw std::invalid_argument("invalid shared config JSON: " + std::string(status.message()));
    }

    options.protocol_version = integer_field(root, "protocol_version");
    options.run_id = string_field(root, "run_id");
    const auto& suffix = object_field(root, "suffix_rpc");
    const auto& coordinator = object_field(root, "coordinator_rpc");
    options.suffix_target = loopback_target(string_field(suffix, "bind"), "suffix_rpc.bind");
    options.coordinator_target = loopback_target(
        string_field(coordinator, "bind"), "coordinator_rpc.bind");
    const auto suffix_limit = integer_field(suffix, "max_message_bytes");
    const auto coordinator_limit = integer_field(coordinator, "max_message_bytes");
    options.max_message_bytes = std::min(suffix_limit, coordinator_limit);
    const double deadline_seconds = std::min(
        number_field(suffix, "deadline_seconds"),
        number_field(coordinator, "deadline_seconds"));
    if (deadline_seconds <= 0.0 || deadline_seconds > 4294967.0) {
        throw std::invalid_argument("shared RPC deadline is outside the client millisecond range");
    }
    options.rpc_deadline_ms = static_cast<std::uint32_t>(deadline_seconds * 1000.0);

    const auto& model = object_field(root, "model");
    options.lora_rank = integer_field(model, "lora_rank");
    options.lora_alpha = static_cast<float>(number_field(model, "lora_alpha"));
    options.learning_rate = static_cast<float>(number_field(model, "learning_rate"));
    if (number_field(model, "lora_dropout") != 0.0) {
        throw std::invalid_argument("Android prefix currently requires model.lora_dropout == 0");
    }

    const auto& training = object_field(root, "training");
    options.total_rounds = integer_field(training, "total_rounds");
    options.cut_layer = integer_field(training, "cut_layer");
    options.local_steps = integer_field(training, "local_steps");
    options.batch_size = integer_field(training, "batch_size");
    options.sequence_length = integer_field(training, "sequence_length");
    options.max_grad_norm = static_cast<float>(number_field(training, "gradient_clip_norm"));
    options.metrics_path = string_field(root, "metrics_path");
}

}  // namespace

std::string client_usage(const char* program) {
    std::ostringstream out;
    out << "Usage: " << program << " [options]\n"
        << "Required:\n"
        << "  --client-id ID              Stable non-empty client identifier\n"
        << "  --config PATH               Shared host/client JSON configuration\n"
        << "  --model-dir PATH            External Llama 3.2 1B asset directory\n"
        << "  --encoder-pte PATH          Exported frozen time-series encoder\n"
        << "  --dataset PATH              Labeled sensor training file\n"
        << "The shared config supplies run ID, endpoints, metrics path, model/LoRA,\n"
        << "training, deadlines, and message limits. Any corresponding CLI option\n"
        << "below overrides it for device-local routing/path needs.\n"
        << "  --run-id ID --suffix HOST:PORT --coordinator HOST:PORT --metrics PATH\n"
        << "Training defaults:\n"
        << "  --cut-layer 1 --local-steps 1 --batch-size 1 --sequence-length 16\n"
        << "  --encoder-input-channels 6 --encoder-input-length 240\n"
        << "  --encoder-output-width 768 --sensor-tokens 162\n"
        << "  --client-index 0 --client-count 1 --lora-rank 8 --lora-alpha 32\n"
        << "  --learning-rate 0.0002 --max-grad-norm 1\n"
        << "Transport defaults:\n"
        << "  --rpc-deadline-ms 30000 --poll-interval-ms 1000 --max-message-bytes 67108864\n";
    return out.str();
}

ClientOptions parse_client_options(int argc, char** argv) {
    std::unordered_map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument("help requested");
        }
        if (arg.rfind("--", 0) != 0) {
            throw std::invalid_argument("unexpected positional argument: " + arg);
        }
        arg.erase(0, 2);
        std::string key;
        std::string value;
        const auto equals = arg.find('=');
        if (equals != std::string::npos) {
            key = arg.substr(0, equals);
            value = arg.substr(equals + 1);
        } else {
            key = arg;
            if (i + 1 >= argc || std::string_view(argv[i + 1]).rfind("--", 0) == 0) {
                throw std::invalid_argument("missing value for --" + key);
            }
            value = argv[++i];
        }
        if (key.empty() || value.empty()) {
            throw std::invalid_argument("empty option name or value");
        }
        if (!values.emplace(key, value).second) {
            throw std::invalid_argument("duplicate option --" + key);
        }
    }

    ClientOptions options;
    const auto config_it = values.find("config");
    if (config_it == values.end()) {
        throw std::invalid_argument("missing --config");
    }
    options.config_path = config_it->second;
    values.erase(config_it);
    apply_shared_config(options.config_path, options);
    auto take_string = [&](const char* key, std::string& target, bool required) {
        const auto it = values.find(key);
        if (it == values.end()) {
            if (required && target.empty()) {
                throw std::invalid_argument(std::string("missing --") + key);
            }
            return;
        }
        target = it->second;
        values.erase(it);
    };
    auto take_u32 = [&](const char* key, std::uint32_t& target) {
        const auto it = values.find(key);
        if (it == values.end()) return;
        target = parse_u32(it->second, key);
        values.erase(it);
    };
    auto take_f32 = [&](const char* key, float& target) {
        const auto it = values.find(key);
        if (it == values.end()) return;
        target = parse_float(it->second, key);
        values.erase(it);
    };

    take_string("client-id", options.client_id, true);
    take_string("run-id", options.run_id, true);
    take_string("model-dir", options.model_dir, true);
    take_string("encoder-pte", options.encoder_program, true);
    take_string("dataset", options.dataset_path, true);
    take_string("suffix", options.suffix_target, true);
    take_string("coordinator", options.coordinator_target, true);
    take_string("metrics", options.metrics_path, true);
    take_u32("protocol-version", options.protocol_version);
    take_u32("cut-layer", options.cut_layer);
    take_u32("local-steps", options.local_steps);
    take_u32("batch-size", options.batch_size);
    take_u32("sequence-length", options.sequence_length);
    take_u32("encoder-input-channels", options.encoder_input_channels);
    take_u32("encoder-input-length", options.encoder_input_length);
    take_u32("encoder-output-width", options.encoder_output_width);
    take_u32("sensor-tokens", options.sensor_tokens);
    take_u32("client-index", options.client_index);
    take_u32("client-count", options.client_count);
    take_u32("lora-rank", options.lora_rank);
    take_f32("lora-alpha", options.lora_alpha);
    take_f32("learning-rate", options.learning_rate);
    take_f32("max-grad-norm", options.max_grad_norm);
    take_u32("rpc-deadline-ms", options.rpc_deadline_ms);
    take_u32("poll-interval-ms", options.poll_interval_ms);
    take_u32("max-message-bytes", options.max_message_bytes);

    if (!values.empty()) {
        throw std::invalid_argument("unknown option --" + values.begin()->first);
    }
    if (options.protocol_version == 0 ||
        options.local_steps == 0 || options.total_rounds == 0 || options.batch_size == 0 ||
        options.sequence_length < 2 || options.client_count == 0 ||
        options.client_index >= options.client_count || options.lora_rank == 0 ||
        options.encoder_input_channels == 0 || options.encoder_input_length == 0 ||
        options.encoder_output_width == 0 ||
        options.sensor_tokens == 0 ||
        options.lora_alpha <= 0.0F || options.learning_rate <= 0.0F ||
        options.max_grad_norm <= 0.0F || options.rpc_deadline_ms == 0 ||
        options.poll_interval_ms == 0 || options.max_message_bytes == 0) {
        throw std::invalid_argument("numeric options are outside their supported positive ranges");
    }
    if (options.cut_layer < 1 || options.cut_layer > 4) {
        throw std::invalid_argument("the mobile runtime supports cut_layer=1..4");
    }
    return options;
}

}  // namespace sflclean
