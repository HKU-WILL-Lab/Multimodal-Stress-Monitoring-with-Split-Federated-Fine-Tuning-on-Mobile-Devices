#include "sfl/lora_state.h"

#include "finetune_ops/nn/lora_linear.h"
#include "sfl/tensor_codec.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace sflclean {
namespace {

void append_linear(std::vector<NamedLoraParameter>& output,
                   int layer,
                   const char* module,
                   const std::unique_ptr<ops::LoRALinear>& linear) {
    if (!linear) return;
    const auto& slices = linear->slices();
    if (slices.size() != 1 || !slices.front().A || !slices.front().B) {
        throw std::logic_error(std::string("expected exactly one LoRA slice for ") + module);
    }
    const std::string base = "model.layers." + std::to_string(layer) + "." + module;
    output.push_back({base + ".lora_A", slices.front().A});
    output.push_back({base + ".lora_B", slices.front().B});
}

}  // namespace

std::vector<NamedLoraParameter> prefix_lora_parameters(ops::GemmaModel& model,
                                                        int cut_layer) {
    if (cut_layer <= 0 || cut_layer > model.config().num_hidden_layers) {
        throw std::invalid_argument("cut layer is outside the decoder");
    }
    std::vector<NamedLoraParameter> parameters;
    for (int layer = 0; layer < cut_layer; ++layer) {
        if (!model.has_layer(layer)) {
            throw std::logic_error("prefix model does not own every layer below the cut");
        }
        const auto& block = model.get_block(layer);
        append_linear(parameters, layer, "self_attn.q_proj", block.q_proj_lora);
        append_linear(parameters, layer, "self_attn.k_proj", block.k_proj_lora);
        append_linear(parameters, layer, "self_attn.v_proj", block.v_proj_lora);
        append_linear(parameters, layer, "self_attn.o_proj", block.o_proj_lora);
        append_linear(parameters, layer, "mlp.gate_proj", block.gate_proj_lora);
        append_linear(parameters, layer, "mlp.up_proj", block.up_proj_lora);
        append_linear(parameters, layer, "mlp.down_proj", block.down_proj_lora);
    }
    std::sort(parameters.begin(), parameters.end(), [](const auto& left, const auto& right) {
        return left.name < right.name;
    });
    const auto duplicate = std::adjacent_find(
        parameters.begin(), parameters.end(), [](const auto& left, const auto& right) {
            return left.name == right.name;
        });
    if (duplicate != parameters.end()) throw std::logic_error("duplicate stable LoRA name");
    return parameters;
}

std::vector<::sfl::clean::v1::Tensor> encode_prefix_lora(
    const std::vector<NamedLoraParameter>& parameters,
    std::size_t byte_limit) {
    std::vector<::sfl::clean::v1::Tensor> result;
    result.reserve(parameters.size());
    for (const auto& parameter : parameters) {
        result.push_back(encode_float32_tensor(parameter.name, parameter.tensor, byte_limit));
    }
    return result;
}

void load_prefix_lora(
    const google::protobuf::RepeatedPtrField<::sfl::clean::v1::Tensor>& wire,
    const std::vector<NamedLoraParameter>& parameters,
    std::size_t byte_limit) {
    if (wire.size() != static_cast<int>(parameters.size())) {
        throw std::invalid_argument("coordinator returned an incomplete prefix LoRA state");
    }
    std::unordered_map<std::string, const ::sfl::clean::v1::Tensor*> by_name;
    for (const auto& item : wire) {
        if (item.name().empty() || !by_name.emplace(item.name(), &item).second) {
            throw std::invalid_argument("coordinator returned an empty or duplicate tensor name");
        }
    }

    std::vector<ops::TensorPtr> decoded;
    decoded.reserve(parameters.size());
    for (const auto& parameter : parameters) {
        const auto found = by_name.find(parameter.name);
        if (found == by_name.end()) {
            throw std::invalid_argument("coordinator omitted prefix tensor " + parameter.name);
        }
        decoded.push_back(decode_float32_tensor(
            *found->second, parameter.name, parameter.tensor->shape(), byte_limit));
    }

    for (std::size_t index = 0; index < parameters.size(); ++index) {
        std::memcpy(parameters[index].tensor->data_ptr(),
                    decoded[index]->data_ptr(),
                    static_cast<std::size_t>(parameters[index].tensor->numel()) * sizeof(float));
    }
}

}  // namespace sflclean
