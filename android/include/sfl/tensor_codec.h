#pragma once

#include "finetune_ops/core/tensor.h"
#include "sfl_clean.pb.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sflclean {

constexpr std::size_t kDefaultTensorByteLimit = 64U * 1024U * 1024U;

::sfl::clean::v1::Tensor encode_float32_tensor(
    const std::string& name,
    const ops::TensorPtr& tensor,
    std::size_t byte_limit = kDefaultTensorByteLimit);

::sfl::clean::v1::Tensor encode_int64_tensor(
    const std::string& name,
    const std::vector<std::int64_t>& values,
    const std::vector<std::int64_t>& shape,
    std::size_t byte_limit = kDefaultTensorByteLimit);

ops::TensorPtr decode_float32_tensor(
    const ::sfl::clean::v1::Tensor& wire,
    const std::string& expected_name,
    const std::vector<std::int64_t>& expected_shape,
    std::size_t byte_limit = kDefaultTensorByteLimit);

std::vector<std::int64_t> decode_int64_tensor(
    const ::sfl::clean::v1::Tensor& wire,
    const std::string& expected_name,
    const std::vector<std::int64_t>& expected_shape,
    std::size_t byte_limit = kDefaultTensorByteLimit);

void validate_finite_scalar(float value, const char* field_name);

}  // namespace sflclean
