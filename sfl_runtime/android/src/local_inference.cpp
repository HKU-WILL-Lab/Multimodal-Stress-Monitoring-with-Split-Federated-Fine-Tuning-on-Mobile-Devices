#include "sfl/local_inference.h"

#include "finetune_ops/core/utils.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "sfl/alignment_projector.h"
#include "sfl/executorch_encoder.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sflclean {
namespace {

using executorch::extension::TensorPtr;
namespace llm = executorch::extension::llm;

constexpr std::int64_t kLlamaWidth = 2048;
constexpr std::int64_t kLlamaVocab = 128256;
constexpr std::uint32_t kSensorMagic = 0x4D574231U;

struct ChannelSpec {
    std::uint8_t kind = 0;
    std::uint8_t value_index = 0;
    float mean = 0.0F;
    float standard_deviation = 1.0F;
};

struct Deployment {
    std::string model_pte;
    std::string tokenizer;
    std::string encoder_pte;
    std::string embedding_dir;
    std::string alignment_checkpoint;
    std::string decoder_method = "forward";
    std::uint32_t encoder_input_length = 0;
    std::uint32_t encoder_output_width = 0;
    std::uint32_t sensor_tokens = 0;
    std::uint32_t max_new_tokens = 96;
    std::vector<ChannelSpec> channels;
};

using Json = nlohmann::json;

const Json& required(const Json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end()) {
        throw std::invalid_argument(std::string("inference config is missing ") + key);
    }
    return *found;
}

std::string string_value(const Json& object, const char* key) {
    const auto& value = required(object, key);
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(std::string("inference config string is invalid: ") + key);
    }
    return value.get<std::string>();
}

std::uint32_t uint_value(const Json& object, const char* key) {
    const auto& value = required(object, key);
    if (!value.is_number()) {
        throw std::invalid_argument(std::string("inference config integer is invalid: ") + key);
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) ||
        number <= 0.0 || number > std::numeric_limits<std::uint32_t>::max() ||
        std::floor(number) != number) {
        throw std::invalid_argument(std::string("inference config integer is invalid: ") + key);
    }
    return static_cast<std::uint32_t>(number);
}

float optional_number(
    const Json& object, const char* key, float fallback) {
    const auto found = object.find(key);
    if (found == object.end()) return fallback;
    if (!found->is_number() || !std::isfinite(found->get<double>())) {
        throw std::invalid_argument(std::string("inference config number is invalid: ") + key);
    }
    return found->get<float>();
}

std::string resolve_asset(const std::filesystem::path& config, const std::string& value) {
    std::filesystem::path path(value);
    if (path.is_relative()) path = config.parent_path() / path;
    path = std::filesystem::weakly_canonical(path);
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("inference asset does not exist: " + path.string());
    }
    return path.string();
}

Deployment read_deployment(const std::string& config_path) {
    const std::filesystem::path path = std::filesystem::canonical(config_path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open inference deployment config");
    std::ostringstream content;
    content << input.rdbuf();
    Json root;
    try {
        root = Json::parse(content.str());
    } catch (const Json::exception&) {
        throw std::invalid_argument("invalid inference deployment JSON");
    }
    if (!root.is_object()) throw std::invalid_argument("inference deployment must be an object");

    Deployment result;
    result.model_pte = resolve_asset(path, string_value(root, "model_pte"));
    result.tokenizer = resolve_asset(path, string_value(root, "tokenizer"));
    result.encoder_pte = resolve_asset(path, string_value(root, "encoder_pte"));
    result.embedding_dir = resolve_asset(path, string_value(root, "embedding_dir"));
    result.alignment_checkpoint = resolve_asset(path, string_value(root, "alignment_checkpoint"));
    const auto decoder = root.find("decoder_method");
    if (decoder != root.end()) {
        if (!decoder->is_string() || decoder->get_ref<const std::string&>().empty()) {
            throw std::invalid_argument("decoder_method must be a non-empty string");
        }
        result.decoder_method = decoder->get<std::string>();
    }
    result.encoder_input_length = uint_value(root, "encoder_input_length");
    result.encoder_output_width = uint_value(root, "encoder_output_width");
    result.sensor_tokens = uint_value(root, "sensor_tokens");
    result.max_new_tokens = uint_value(root, "max_new_tokens");

    const auto& channel_value = required(root, "channels");
    if (!channel_value.is_array() || channel_value.empty()) {
        throw std::invalid_argument("channels must be a non-empty array");
    }
    for (const auto& channel : channel_value) {
        if (!channel.is_object()) {
            throw std::invalid_argument("each channel must be an object");
        }
        const auto kind = uint_value(channel, "kind");
        const auto index = optional_number(channel, "value_index", 0.0F);
        const float deviation = optional_number(channel, "std", 1.0F);
        if (kind > 255 || index < 0.0F || index > 7.0F || std::floor(index) != index ||
            deviation <= 0.0F) {
            throw std::invalid_argument("channel selector or normalization is invalid");
        }
        result.channels.push_back(ChannelSpec{
            static_cast<std::uint8_t>(kind), static_cast<std::uint8_t>(index),
            optional_number(channel, "mean", 0.0F), deviation});
    }
    return result;
}

std::uint32_t read_u32_be(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    if (offset + 4 > data.size()) throw std::invalid_argument("truncated sensor window");
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value = (value << 8U) | data[offset++];
    return value;
}

std::uint64_t read_u64_be(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    if (offset + 8 > data.size()) throw std::invalid_argument("truncated sensor window");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value = (value << 8U) | data[offset++];
    return value;
}

float read_f32_be(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    const std::uint32_t bits = read_u32_be(data, offset);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) throw std::invalid_argument("sensor window contains a non-finite value");
    return value;
}

struct Sample {
    std::uint64_t timestamp = 0;
    std::uint8_t kind = 0;
    std::vector<float> values;
};

std::vector<float> preprocess_window(
    const std::vector<std::uint8_t>& bytes, const Deployment& deployment) {
    std::size_t offset = 0;
    if (read_u32_be(bytes, offset) != kSensorMagic) {
        throw std::invalid_argument("invalid sensor window magic");
    }
    const std::uint64_t started = read_u64_be(bytes, offset);
    const std::uint64_t ended = read_u64_be(bytes, offset);
    const std::uint32_t count = read_u32_be(bytes, offset);
    if (ended < started || count == 0 || count > 10000) {
        throw std::invalid_argument("invalid or empty sensor window");
    }
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        Sample sample;
        sample.timestamp = read_u64_be(bytes, offset);
        if (offset + 2 > bytes.size()) throw std::invalid_argument("truncated sensor sample");
        sample.kind = bytes[offset++];
        const std::uint8_t values = bytes[offset++];
        if (values == 0 || values > 8) throw std::invalid_argument("invalid sensor value count");
        sample.values.reserve(values);
        for (std::uint8_t j = 0; j < values; ++j) sample.values.push_back(read_f32_be(bytes, offset));
        samples.push_back(std::move(sample));
    }
    if (offset != bytes.size()) throw std::invalid_argument("unexpected bytes after sensor window");

    std::vector<float> output;
    output.reserve(deployment.channels.size() * deployment.encoder_input_length);
    for (const auto& channel : deployment.channels) {
        std::vector<std::pair<std::uint64_t, float>> points;
        for (const auto& sample : samples) {
            if (sample.kind == channel.kind && channel.value_index < sample.values.size()) {
                points.emplace_back(sample.timestamp, sample.values[channel.value_index]);
            }
        }
        if (points.empty()) throw std::invalid_argument("a configured sensor channel has no samples");
        std::sort(points.begin(), points.end());
        std::size_t upper = 0;
        for (std::uint32_t i = 0; i < deployment.encoder_input_length; ++i) {
            const double fraction = deployment.encoder_input_length == 1
                ? 0.0 : static_cast<double>(i) / (deployment.encoder_input_length - 1);
            const std::uint64_t timestamp = started + static_cast<std::uint64_t>((ended - started) * fraction);
            while (upper < points.size() && points[upper].first < timestamp) ++upper;
            float value = 0.0F;
            if (upper == 0) value = points.front().second;
            else if (upper == points.size()) value = points.back().second;
            else {
                const auto& left = points[upper - 1];
                const auto& right = points[upper];
                const double span = static_cast<double>(right.first - left.first);
                const double alpha = span == 0.0 ? 0.0 : (timestamp - left.first) / span;
                value = static_cast<float>(left.second + (right.second - left.second) * alpha);
            }
            output.push_back((value - channel.mean) / channel.standard_deviation);
        }
    }
    return output;
}

ops::TensorPtr load_embedding(const std::string& model_dir) {
    ops::SafeTensorsModelReader reader(model_dir);
    reader.parse_headers();
    ops::SafeTensorsLoadOptions options;
    options.preserve_low_precision_key_substrings = {"embed_tokens"};
    auto tensors = reader.load_tensors_mapped(
        {{"embed_tokens.weight", "model.embed_tokens.weight"}}, options);
    const auto found = tensors.find("embed_tokens.weight");
    if (found == tensors.end() || found->second->shape() !=
        std::vector<std::int64_t>{kLlamaVocab, kLlamaWidth}) {
        throw std::invalid_argument("invalid Llama 3.2 1B embedding asset");
    }
    return found->second;
}

float weight_value(const ops::TensorPtr& tensor, std::int64_t index) {
    if (tensor->dtype() == ops::kFloat32) return tensor->data<float>()[index];
    if (tensor->dtype() == ops::kFloat16) return ops::fp16_bits_to_float32(tensor->data<std::uint16_t>()[index]);
    if (tensor->dtype() == ops::kBFloat16) return ops::bf16_bits_to_float32(tensor->data<std::uint16_t>()[index]);
    throw std::invalid_argument("unsupported Llama embedding dtype");
}

std::vector<float> embed_tokens(
    const ops::TensorPtr& table, const std::vector<std::uint64_t>& tokens) {
    std::vector<float> output(tokens.size() * kLlamaWidth);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        if (tokens[row] >= static_cast<std::uint64_t>(kLlamaVocab)) {
            throw std::invalid_argument("token id exceeds Llama vocabulary");
        }
        const std::int64_t start = static_cast<std::int64_t>(tokens[row]) * kLlamaWidth;
        for (std::int64_t column = 0; column < kLlamaWidth; ++column) {
            output[row * kLlamaWidth + column] = weight_value(table, start + column);
        }
    }
    return output;
}

void load_alignment(const std::string& directory, AlignmentProjector& projector) {
    const auto root = std::filesystem::canonical(directory);
    std::ifstream input(root / "manifest.json", std::ios::binary);
    if (!input) throw std::invalid_argument("alignment checkpoint has no manifest.json");
    std::ostringstream content;
    content << input.rdbuf();
    Json manifest;
    try {
        manifest = Json::parse(content.str());
    } catch (const Json::exception&) {
        throw std::invalid_argument("alignment checkpoint manifest is invalid");
    }
    const auto& entries = required(manifest, "tensors");
    if (!entries.is_array()) {
        throw std::invalid_argument("alignment checkpoint tensors must be an array");
    }
    auto parameters = projector.named_parameters();
    for (auto& parameter : parameters) {
        const Json* selected = nullptr;
        for (const auto& entry : entries) {
            if (!entry.is_object()) continue;
            const auto name = entry.find("name");
            if (name != entry.end() && name->is_string() && name->get<std::string>() == parameter.name) {
                selected = &entry;
                break;
            }
        }
        if (selected == nullptr) throw std::invalid_argument("alignment checkpoint is missing " + parameter.name);
        const auto filename = string_value(*selected, "file");
        const auto file = std::filesystem::weakly_canonical(root / filename);
        if (file.parent_path() != root) throw std::invalid_argument("alignment tensor path escapes checkpoint");
        std::ifstream tensor(file, std::ios::binary | std::ios::ate);
        const auto expected = static_cast<std::streamsize>(parameter.tensor->numel() * sizeof(float));
        if (!tensor || tensor.tellg() != expected) throw std::invalid_argument("alignment tensor size mismatch");
        tensor.seekg(0);
        tensor.read(reinterpret_cast<char*>(parameter.tensor->data<float>()), expected);
        if (!tensor) throw std::invalid_argument("cannot read alignment tensor");
    }
}

int parse_score(const std::string& text, const char* label) {
    const std::regex pattern(std::string(label) + R"(\s*:\s*([1-5]))", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("model output is missing ") + label);
    }
    return std::stoi(match[1].str());
}

InferenceResult parse_result(const std::string& text) {
    InferenceResult result;
    result.valence = parse_score(text, "Valence");
    result.arousal = parse_score(text, "Arousal");
    const std::regex assessment(R"(Assessment\s*:\s*(.+))", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(text, match, assessment) || match[1].str().empty()) {
        throw std::runtime_error("model output is missing Assessment");
    }
    result.assessment = match[1].str();
    result.generated_text = text;
    return result;
}

}  // namespace

struct LocalInference::Impl {
    explicit Impl(const std::string& config)
        : deployment(read_deployment(config)),
          encoder(deployment.encoder_pte, deployment.channels.size(),
                  deployment.encoder_input_length, deployment.sensor_tokens,
                  deployment.encoder_output_width),
          projector(deployment.encoder_output_width, kLlamaWidth),
          embedding(load_embedding(deployment.embedding_dir)),
          decoder(deployment.model_pte),
          tokenizer(llm::load_tokenizer(deployment.tokenizer)) {
        load_alignment(deployment.alignment_checkpoint, projector);
        if (!tokenizer) throw std::runtime_error("failed to load Llama tokenizer");
        if (decoder.load_method(deployment.decoder_method) != executorch::runtime::Error::Ok) {
            throw std::runtime_error("failed to load Llama decoder method");
        }
        auto meta = decoder.method_meta(deployment.decoder_method);
        if (!meta.ok() || meta->num_inputs() < 1 || meta->num_inputs() > 2) {
            throw std::runtime_error("decoder must accept embeddings and optional cache position");
        }
        const auto embedding_meta = meta->input_tensor_meta(0);
        if (!embedding_meta.ok() ||
            embedding_meta->scalar_type() != executorch::aten::ScalarType::Float ||
            embedding_meta->sizes().size() != 3 ||
            embedding_meta->sizes()[2] != kLlamaWidth) {
            throw std::runtime_error(
                "decoder input 0 must be float32 Llama embeddings with width 2048");
        }
        uses_kv_cache = meta->num_inputs() == 2;
        if (uses_kv_cache) {
            const auto position_meta = meta->input_tensor_meta(1);
            if (!position_meta.ok() ||
                position_meta->scalar_type() != executorch::aten::ScalarType::Long) {
                throw std::runtime_error("decoder input 1 must be an int64 cache position");
            }
        }
    }

    executorch::aten::Tensor execute(
        std::vector<float>& values, std::int64_t sequence, std::int64_t start_position) {
        auto input = executorch::extension::from_blob(
            values.data(),
            {1, static_cast<int>(sequence), static_cast<int>(kLlamaWidth)},
            executorch::aten::ScalarType::Float);
        std::vector<executorch::runtime::EValue> arguments{input};
        std::vector<std::int64_t> positions;
        TensorPtr position;
        if (uses_kv_cache) {
            auto position_result = llm::populate_start_pos_or_cache_position(
                &decoder, start_position, positions, static_cast<int>(sequence),
                deployment.decoder_method.c_str());
            if (!position_result.ok()) throw std::runtime_error("cannot create decoder cache position");
            position = *position_result;
            arguments.emplace_back(position);
        }
        auto output = decoder.execute(deployment.decoder_method, arguments);
        if (!output.ok() || output->empty() || !output->front().isTensor()) {
            throw std::runtime_error("Llama decoder execution failed");
        }
        return output->front().toTensor();
    }

    std::uint64_t greedy(const executorch::aten::Tensor& logits) const {
        if (logits.scalar_type() != executorch::aten::ScalarType::Float ||
            logits.numel() < kLlamaVocab) {
            throw std::runtime_error("decoder returned invalid logits");
        }
        const float* values = logits.const_data_ptr<float>() + logits.numel() - kLlamaVocab;
        return static_cast<std::uint64_t>(
            std::max_element(values, values + kLlamaVocab) - values);
    }

    InferenceResult infer(const std::vector<std::uint8_t>& bytes) {
        auto window = preprocess_window(bytes, deployment);
        auto features = encoder.encode(window);
        auto feature_tensor = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{1, deployment.sensor_tokens,
                                      deployment.encoder_output_width},
            features.data(), ops::kFloat32, ops::kCPU);
        auto sensor = projector.forward(feature_tensor);

        const std::string prompt_prefix =
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Given the wearable-sensor representation, output valence and arousal as "
            "integers from 1 to 5, followed by a concise assessment.";
        const std::string prompt_suffix =
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        auto prefix_ids = tokenizer->encode(prompt_prefix, 0, 0);
        auto suffix_ids = tokenizer->encode(prompt_suffix, 0, 0);
        if (!prefix_ids.ok() || prefix_ids->empty() ||
            !suffix_ids.ok() || suffix_ids->empty()) {
            throw std::runtime_error("cannot tokenize inference prompt");
        }
        auto prefix = embed_tokens(embedding, *prefix_ids);
        auto suffix = embed_tokens(embedding, *suffix_ids);
        std::vector<float> all;
        const auto prompt_tokens = prefix_ids->size() + suffix_ids->size();
        all.reserve((deployment.sensor_tokens + prompt_tokens + deployment.max_new_tokens) * kLlamaWidth);
        all.insert(all.end(), prefix.begin(), prefix.end());
        all.insert(all.end(), sensor->data<float>(), sensor->data<float>() + sensor->numel());
        all.insert(all.end(), suffix.begin(), suffix.end());
        std::int64_t occupied = deployment.sensor_tokens +
            static_cast<std::int64_t>(prompt_tokens);
        auto logits = execute(all, occupied, 0);

        std::string generated;
        std::uint64_t previous = suffix_ids->back();
        for (std::uint32_t i = 0; i < deployment.max_new_tokens; ++i) {
            const std::uint64_t token = greedy(logits);
            if (token == tokenizer->eos_tok() || token == 128001 || token == 128009) break;
            auto piece = tokenizer->decode(previous, token, true);
            if (!piece.ok()) throw std::runtime_error("cannot decode generated token");
            generated += *piece;
            const std::vector<std::uint64_t> one{token};
            auto next_embedding = embed_tokens(embedding, one);
            if (uses_kv_cache) {
                logits = execute(next_embedding, 1, occupied);
            } else {
                all.insert(all.end(), next_embedding.begin(), next_embedding.end());
                logits = execute(all, occupied + 1, 0);
            }
            ++occupied;
            previous = token;
        }
        return parse_result(generated);
    }

    Deployment deployment;
    ExecuTorchEncoder encoder;
    AlignmentProjector projector;
    ops::TensorPtr embedding;
    executorch::extension::Module decoder;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    bool uses_kv_cache = false;
};

LocalInference::LocalInference(const std::string& config)
    : impl_(std::make_unique<Impl>(config)) {}

LocalInference::~LocalInference() = default;

InferenceResult LocalInference::infer(const std::vector<std::uint8_t>& sensor_window) {
    return impl_->infer(sensor_window);
}

}  // namespace sflclean
