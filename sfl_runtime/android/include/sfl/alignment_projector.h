#pragma once

#include "finetune_ops/core/tensor.h"
#include "sfl/named_parameter.h"

#include <cstdint>
#include <vector>

namespace sflclean {

class AlignmentProjector {
public:
    AlignmentProjector(std::int64_t encoder_width,
                       std::int64_t llm_width,
                       std::uint64_t seed = 42);

    // OpenTSLM-SP alignment: LayerNorm(encoder_width), Linear to the LLM
    // width, then GELU, independently for every encoder token.
    // features: [batch, sensor_tokens, encoder_width]
    ops::TensorPtr forward(const ops::TensorPtr& features) const;
    std::vector<NamedLoraParameter> named_parameters() const;

private:
    std::int64_t encoder_width_;
    ops::TensorPtr norm_weight_;
    ops::TensorPtr norm_bias_;
    ops::TensorPtr projection_weight_;
    ops::TensorPtr projection_bias_;
};

}  // namespace sflclean
