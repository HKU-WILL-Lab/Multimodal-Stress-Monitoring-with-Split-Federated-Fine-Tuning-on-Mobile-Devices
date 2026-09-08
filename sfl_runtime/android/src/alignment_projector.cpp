#include "sfl/alignment_projector.h"

#include "finetune_ops/core/ops.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace sflclean {
namespace {

void initialize_weight(const ops::TensorPtr& tensor, std::int64_t fan_in,
                       std::mt19937& generator) {
    const float bound = std::sqrt(6.0F / static_cast<float>(fan_in));
    std::uniform_real_distribution<float> distribution(-bound, bound);
    float* values = tensor->data<float>();
    for (std::int64_t i = 0; i < tensor->numel(); ++i) values[i] = distribution(generator);
    tensor->set_requires_grad(true);
}

ops::TensorPtr add_bias(const ops::TensorPtr& value, const ops::TensorPtr& bias) {
    const auto& shape = value->shape();
    if (shape.size() < 2 || bias->shape() != std::vector<std::int64_t>{shape.back()}) {
        throw std::logic_error("alignment bias shape mismatch");
    }
    // The framework's add operator reduces the broadcast gradient back to bias.
    return ops::add(value, bias);
}

}  // namespace

AlignmentProjector::AlignmentProjector(std::int64_t encoder_width,
                                       std::int64_t llm_width,
                                       std::uint64_t seed)
    : encoder_width_(encoder_width) {
    if (encoder_width <= 0 || llm_width <= 0) {
        throw std::invalid_argument("alignment dimensions must be positive");
    }
    norm_weight_ = ops::full({encoder_width}, 1.0F);
    norm_bias_ = ops::zeros({encoder_width});
    projection_weight_ = ops::zeros({encoder_width, llm_width});
    projection_bias_ = ops::zeros({llm_width});
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    initialize_weight(projection_weight_, encoder_width, generator);
    norm_weight_->set_requires_grad(true);
    norm_bias_->set_requires_grad(true);
    projection_bias_->set_requires_grad(true);
}

ops::TensorPtr AlignmentProjector::forward(const ops::TensorPtr& features) const {
    if (!features || features->dtype() != ops::kFloat32 ||
        features->shape().size() != 3 || features->shape()[2] != encoder_width_) {
        throw std::invalid_argument("alignment input shape or dtype is invalid");
    }
    auto normalized = ops::layer_norm(features, norm_weight_, norm_bias_, 1.0e-5F);
    return ops::gelu(add_bias(
        ops::matmul(normalized, projection_weight_), projection_bias_));
}

std::vector<NamedLoraParameter> AlignmentProjector::named_parameters() const {
    return {
        {"alignment.layers.0.bias", norm_bias_},
        {"alignment.layers.0.weight", norm_weight_},
        {"alignment.layers.1.bias", projection_bias_},
        {"alignment.layers.1.weight", projection_weight_},
    };
}

}  // namespace sflclean
