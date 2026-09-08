#include "sfl/llama_prefix_block.h"

#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/backward_functions.h"
#include "finetune_ops/graph/safetensors_loader.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace sflclean {
namespace {

constexpr std::int64_t kHidden = 2048;
constexpr std::int64_t kIntermediate = 8192;
constexpr std::int64_t kHeads = 32;
constexpr std::int64_t kKvHeads = 8;
constexpr std::int64_t kHeadDim = 64;
constexpr float kRmsEps = 1.0e-5F;
constexpr float kRopeTheta = 500000.0F;
constexpr float kMaskValue = -1.0e9F;
constexpr float kRopeFactor = 32.0F;
constexpr float kOriginalContext = 8192.0F;
constexpr float kLowFrequencyFactor = 1.0F;
constexpr float kHighFrequencyFactor = 4.0F;
constexpr float kPi = 3.14159265358979323846F;

float llama3_frequency(std::int64_t dimension) {
    const float base_frequency = 1.0F / std::pow(
        kRopeTheta, 2.0F * static_cast<float>(dimension) / static_cast<float>(kHeadDim));
    const float wavelength = 2.0F * kPi / base_frequency;
    const float low_wavelength = kOriginalContext / kLowFrequencyFactor;
    const float high_wavelength = kOriginalContext / kHighFrequencyFactor;
    if (wavelength < high_wavelength) return base_frequency;
    if (wavelength > low_wavelength) return base_frequency / kRopeFactor;
    const float smooth = (kOriginalContext / wavelength - kLowFrequencyFactor) /
                         (kHighFrequencyFactor - kLowFrequencyFactor);
    return (1.0F - smooth) * base_frequency / kRopeFactor + smooth * base_frequency;
}

ops::TensorPtr rotate_llama3(const ops::TensorPtr& input, bool inverse) {
    const auto& shape = input->shape();
    if (shape.size() != 4 || shape[3] != kHeadDim) {
        throw std::invalid_argument("Llama3 RoPE expects [batch, heads, sequence, 64]");
    }
    auto result = ops::zeros(shape);
    const auto* source = input->data<float>();
    auto* target = result->data<float>();
    const auto half = kHeadDim / 2;
    for (std::int64_t b = 0; b < shape[0]; ++b) {
        for (std::int64_t h = 0; h < shape[1]; ++h) {
            for (std::int64_t position = 0; position < shape[2]; ++position) {
                const auto base = ((b * shape[1] + h) * shape[2] + position) * kHeadDim;
                for (std::int64_t d = 0; d < half; ++d) {
                    const float angle = static_cast<float>(position) * llama3_frequency(d);
                    const float cosine = std::cos(angle);
                    const float sine = inverse ? -std::sin(angle) : std::sin(angle);
                    const float first = source[base + d];
                    const float second = source[base + half + d];
                    target[base + d] = first * cosine - second * sine;
                    target[base + half + d] = first * sine + second * cosine;
                }
            }
        }
    }
    return result;
}

class Llama3RopeBackward final : public ops::BackwardFunction {
public:
    std::vector<ops::TensorPtr> apply(const ops::TensorPtr& gradient) override {
        return {rotate_llama3(gradient, true)};
    }
};

ops::TensorPtr apply_llama3_rope(const ops::TensorPtr& input) {
    auto result = rotate_llama3(input, false);
    if (input->requires_grad()) {
        result->set_requires_grad(true);
#ifdef USE_NEW_AUTOGRAD_ENGINE
        ops::autograd::Engine::instance().register_node(
            result, {input}, std::make_shared<Llama3RopeBackward>());
#else
        result->set_grad_fn([input](const ops::TensorPtr& gradient) {
            auto value = rotate_llama3(gradient, true);
            input->set_grad(value);
            return std::vector<ops::TensorPtr>{value};
        });
#endif
    }
    return result;
}

ops::TensorPtr required(
    const std::unordered_map<std::string, ops::TensorPtr>& tensors,
    const char* name, const std::vector<std::int64_t>& shape) {
    const auto found = tensors.find(name);
    if (found == tensors.end() || found->second->shape() != shape) {
        throw std::invalid_argument(std::string("missing or incompatible Llama block-0 weight: ") + name);
    }
    return found->second;
}

std::pair<ops::TensorPtr, ops::TensorPtr> make_lora(
    std::int64_t input, std::int64_t output, std::uint32_t rank,
    std::mt19937& generator) {
    const float bound = 1.0F / std::sqrt(static_cast<float>(input));
    std::uniform_real_distribution<float> distribution(-bound, bound);
    auto a = ops::zeros({static_cast<std::int64_t>(rank), input});
    auto b = ops::zeros({output, static_cast<std::int64_t>(rank)});
    for (std::int64_t i = 0; i < a->numel(); ++i) a->data<float>()[i] = distribution(generator);
    a->set_requires_grad(true);
    b->set_requires_grad(true);
    return {a, b};
}

void attach(ops::LoRALinear& linear, std::int64_t input, std::int64_t output,
            std::uint32_t rank, float scale, std::mt19937& generator,
            const char* debug_name) {
    auto [a, b] = make_lora(input, output, rank, generator);
    linear.attach_lora(a, b, scale, 0, static_cast<int>(output));
    linear.set_debug_name(debug_name);
}

ops::TensorPtr causal_padding_mask(const ops::TensorPtr& attention_mask) {
    const auto& shape = attention_mask->shape();
    if (attention_mask->dtype() != ops::kInt32 || shape.size() != 2) {
        throw std::invalid_argument("Llama attention mask must be int32 [batch, sequence]");
    }
    const auto batch = shape[0];
    const auto sequence = shape[1];
    auto mask = ops::zeros({batch, 1, sequence, sequence});
    auto* output = mask->data<float>();
    const auto* visible = attention_mask->data<std::int32_t>();
    for (std::int64_t row = 0; row < batch; ++row) {
        for (std::int64_t query = 0; query < sequence; ++query) {
            for (std::int64_t key = 0; key < sequence; ++key) {
                if (key > query || visible[row * sequence + key] == 0) {
                    output[(row * sequence + query) * sequence + key] = kMaskValue;
                }
            }
        }
    }
    return mask;
}

void append_named(std::vector<NamedLoraParameter>& output, const char* module,
                  const std::unique_ptr<ops::LoRALinear>& linear) {
    if (!linear || linear->slices().size() != 1) {
        throw std::logic_error("Llama prefix LoRA module is not initialized");
    }
    output.push_back({std::string("layers.0.") + module + ".lora_A", linear->slices()[0].A});
    output.push_back({std::string("layers.0.") + module + ".lora_B", linear->slices()[0].B});
}

}  // namespace

LlamaPrefixBlock::LlamaPrefixBlock(const std::string& model_dir,
                                   std::uint32_t lora_rank,
                                   float lora_alpha,
                                   std::uint64_t seed) {
    if (lora_rank == 0 || !std::isfinite(lora_alpha) || lora_alpha <= 0.0F) {
        throw std::invalid_argument("Llama prefix LoRA configuration is invalid");
    }
    const std::string prefix = "model.layers.0.";
    std::unordered_map<std::string, std::string> mapping = {
        {"input_norm", prefix + "input_layernorm.weight"},
        {"post_attention_norm", prefix + "post_attention_layernorm.weight"},
        {"q", prefix + "self_attn.q_proj.weight"},
        {"k", prefix + "self_attn.k_proj.weight"},
        {"v", prefix + "self_attn.v_proj.weight"},
        {"o", prefix + "self_attn.o_proj.weight"},
        {"gate", prefix + "mlp.gate_proj.weight"},
        {"up", prefix + "mlp.up_proj.weight"},
        {"down", prefix + "mlp.down_proj.weight"},
    };
    ops::SafeTensorsModelReader reader(model_dir);
    reader.parse_headers();
    ops::SafeTensorsLoadOptions options;
    options.verbose = false;
    const auto loaded = reader.load_tensors_mapped(mapping, options);
    weights_.input_norm = required(loaded, "input_norm", {kHidden});
    weights_.post_attention_norm = required(loaded, "post_attention_norm", {kHidden});
    weights_.q = required(loaded, "q", {kHidden, kHidden});
    weights_.k = required(loaded, "k", {kHidden, kKvHeads * kHeadDim});
    weights_.v = required(loaded, "v", {kHidden, kKvHeads * kHeadDim});
    weights_.o = required(loaded, "o", {kHidden, kHidden});
    weights_.gate = required(loaded, "gate", {kHidden, kIntermediate});
    weights_.up = required(loaded, "up", {kHidden, kIntermediate});
    weights_.down = required(loaded, "down", {kIntermediate, kHidden});

    q_ = std::make_unique<ops::LoRALinear>(weights_.q);
    k_ = std::make_unique<ops::LoRALinear>(weights_.k);
    v_ = std::make_unique<ops::LoRALinear>(weights_.v);
    o_ = std::make_unique<ops::LoRALinear>(weights_.o);
    gate_ = std::make_unique<ops::LoRALinear>(weights_.gate);
    up_ = std::make_unique<ops::LoRALinear>(weights_.up);
    down_ = std::make_unique<ops::LoRALinear>(weights_.down);
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    const float scale = lora_alpha / static_cast<float>(lora_rank);
    attach(*q_, kHidden, kHidden, lora_rank, scale, generator, "layers_0_q_proj");
    attach(*k_, kHidden, kKvHeads * kHeadDim, lora_rank, scale, generator, "layers_0_k_proj");
    attach(*v_, kHidden, kKvHeads * kHeadDim, lora_rank, scale, generator, "layers_0_v_proj");
    attach(*o_, kHidden, kHidden, lora_rank, scale, generator, "layers_0_o_proj");
    attach(*gate_, kHidden, kIntermediate, lora_rank, scale, generator, "layers_0_gate_proj");
    attach(*up_, kHidden, kIntermediate, lora_rank, scale, generator, "layers_0_up_proj");
    attach(*down_, kIntermediate, kHidden, lora_rank, scale, generator, "layers_0_down_proj");
}

ops::TensorPtr LlamaPrefixBlock::forward(
    const ops::TensorPtr& hidden_states,
    const ops::TensorPtr& attention_mask) const {
    if (!hidden_states || hidden_states->dtype() != ops::kFloat32 ||
        hidden_states->shape().size() != 3 || hidden_states->shape()[2] != kHidden ||
        attention_mask->shape() != std::vector<std::int64_t>{
            hidden_states->shape()[0], hidden_states->shape()[1]}) {
        throw std::invalid_argument("Llama prefix input shape is invalid");
    }
    const auto batch = hidden_states->shape()[0];
    const auto sequence = hidden_states->shape()[1];
    auto normalized = ops::rms_norm_affine(hidden_states, weights_.input_norm, kRmsEps);
    auto q = ops::permute(ops::reshape(q_->forward(normalized),
        {batch, sequence, kHeads, kHeadDim}), {0, 2, 1, 3});
    auto k = ops::permute(ops::reshape(k_->forward(normalized),
        {batch, sequence, kKvHeads, kHeadDim}), {0, 2, 1, 3});
    auto v = ops::permute(ops::reshape(v_->forward(normalized),
        {batch, sequence, kKvHeads, kHeadDim}), {0, 2, 1, 3});
    q = apply_llama3_rope(q);
    k = apply_llama3_rope(k);
    k = ops::repeat_kv_heads(k, kHeads / kKvHeads);
    v = ops::repeat_kv_heads(v, kHeads / kKvHeads);
    auto scores = ops::mul(ops::matmul(q, ops::transpose(k, 2, 3)),
                           1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    scores = ops::add(scores, causal_padding_mask(attention_mask));
    auto context = ops::matmul(ops::softmax(scores, -1), v);
    context = ops::reshape(ops::permute(context, {0, 2, 1, 3}),
                           {batch, sequence, kHidden});
    auto after_attention = ops::add(hidden_states, o_->forward(context));
    normalized = ops::rms_norm_affine(
        after_attention, weights_.post_attention_norm, kRmsEps);
    auto mlp = down_->forward(ops::mul(
        ops::silu(gate_->forward(normalized)), up_->forward(normalized)));
    return ops::add(after_attention, mlp);
}

std::vector<NamedLoraParameter> LlamaPrefixBlock::named_parameters() const {
    std::vector<NamedLoraParameter> result;
    result.reserve(14);
    append_named(result, "q_proj", q_);
    append_named(result, "k_proj", k_);
    append_named(result, "v_proj", v_);
    append_named(result, "o_proj", o_);
    append_named(result, "gate_proj", gate_);
    append_named(result, "up_proj", up_);
    append_named(result, "down_proj", down_);
    return result;
}

}  // namespace sflclean
