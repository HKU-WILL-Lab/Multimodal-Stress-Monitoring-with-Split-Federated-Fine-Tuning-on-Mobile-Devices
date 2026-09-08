#pragma once

#include "finetune_ops/core/tensor.h"

#include <string>

namespace sflclean {

// Pure model-state descriptor shared by the training and local-inference
// runtimes. Wire-format concerns belong in lora_state.h instead.
struct NamedLoraParameter {
    std::string name;
    ops::TensorPtr tensor;
};

}  // namespace sflclean
