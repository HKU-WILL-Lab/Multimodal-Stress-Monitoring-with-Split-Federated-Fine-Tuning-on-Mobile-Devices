#pragma once

#include "finetune_ops/core/tensor.h"
#include "finetune_ops/nn/lora_linear.h"
#include "sfl/named_parameter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sflclean {

// One indexed decoder block of Llama 3.2 1B. Frozen base weights are read from
// the Hugging Face checkpoint; only its LoRA branches are trainable.
class LlamaPrefixBlock {
public:
    LlamaPrefixBlock(const std::string& model_dir,
                     std::uint32_t lora_rank,
                     float lora_alpha,
                     std::uint64_t seed = 42,
                     std::uint32_t layer_index = 0);

    // hidden_states: [batch, sequence, 2048]
    // attention_mask: [batch, sequence], int32, where one means visible.
    ops::TensorPtr forward(const ops::TensorPtr& hidden_states,
                           const ops::TensorPtr& attention_mask) const;
    std::vector<NamedLoraParameter> named_parameters() const;

private:
    std::uint32_t layer_index_;
    struct Weights {
        ops::TensorPtr input_norm;
        ops::TensorPtr post_attention_norm;
        ops::TensorPtr q;
        ops::TensorPtr k;
        ops::TensorPtr v;
        ops::TensorPtr o;
        ops::TensorPtr gate;
        ops::TensorPtr up;
        ops::TensorPtr down;
    } weights_;

    std::unique_ptr<ops::LoRALinear> q_;
    std::unique_ptr<ops::LoRALinear> k_;
    std::unique_ptr<ops::LoRALinear> v_;
    std::unique_ptr<ops::LoRALinear> o_;
    std::unique_ptr<ops::LoRALinear> gate_;
    std::unique_ptr<ops::LoRALinear> up_;
    std::unique_ptr<ops::LoRALinear> down_;
};

}  // namespace sflclean
