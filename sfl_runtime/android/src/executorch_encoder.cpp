#include "sfl/executorch_encoder.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sflclean {
namespace {

int32_t checked_dimension(std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(std::string(name) + " is outside ExecuTorch's dimension range");
    }
    return static_cast<int32_t>(value);
}

}  // namespace

struct ExecuTorchEncoder::Impl {
    explicit Impl(const std::string& path) : module(path) {
        if (module.load() != executorch::runtime::Error::Ok) {
            throw std::runtime_error("failed to load ExecuTorch encoder program: " + path);
        }
    }
    executorch::extension::Module module;
};

ExecuTorchEncoder::ExecuTorchEncoder(std::string program_path,
                                     std::size_t input_channels,
                                     std::size_t input_length,
                                     std::size_t output_tokens,
                                     std::size_t output_width)
    : impl_(std::make_unique<Impl>(program_path)),
      input_channels_(input_channels),
      input_length_(input_length),
      output_tokens_(output_tokens),
      output_width_(output_width) {
    checked_dimension(input_channels_, "input_channels");
    checked_dimension(input_length_, "input_length");
    checked_dimension(output_width_, "output_width");
    checked_dimension(output_tokens_, "output_tokens");
}

ExecuTorchEncoder::~ExecuTorchEncoder() = default;

std::vector<float> ExecuTorchEncoder::encode(const std::vector<float>& window) {
    if (window.size() != input_size()) {
        throw std::invalid_argument(
            "encoder input contains " + std::to_string(window.size()) +
            " values; deployment manifest requires " + std::to_string(input_size()));
    }
    auto input = executorch::extension::from_blob(
        const_cast<float*>(window.data()),
        {1, checked_dimension(input_channels_, "input_channels"),
         checked_dimension(input_length_, "input_length")});
    const auto result = impl_->module.forward(input);
    if (!result.ok() || result->empty() || !result->at(0).isTensor()) {
        throw std::runtime_error("ExecuTorch encoder forward failed or returned a non-tensor");
    }
    const auto output = result->at(0).toTensor();
    if (output.scalar_type() != executorch::aten::ScalarType::Float ||
        output.numel() != static_cast<std::int64_t>(output_tokens_ * output_width_)) {
        throw std::runtime_error("ExecuTorch encoder output does not match deployment manifest");
    }
    const float* data = output.const_data_ptr<float>();
    return std::vector<float>(data, data + output_tokens_ * output_width_);
}

}  // namespace sflclean
