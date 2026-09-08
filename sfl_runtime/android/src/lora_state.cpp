#include "sfl/lora_state.h"

#include "sfl/tensor_codec.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace sflclean {
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
