#pragma once

#include "finetune_ops/core/tensor.h"
#include "sfl/named_parameter.h"
#include "sfl_clean.pb.h"

#include <cstddef>
#include <string>
#include <vector>

namespace sflclean {

std::vector<::sfl::clean::v1::Tensor> encode_prefix_lora(
    const std::vector<NamedLoraParameter>& parameters,
    std::size_t byte_limit);

void load_prefix_lora(
    const google::protobuf::RepeatedPtrField<::sfl::clean::v1::Tensor>& wire,
    const std::vector<NamedLoraParameter>& parameters,
    std::size_t byte_limit);

}  // namespace sflclean
