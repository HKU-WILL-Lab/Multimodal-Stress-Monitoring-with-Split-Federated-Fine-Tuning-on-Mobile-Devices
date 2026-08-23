#include "sfl/tensor_codec.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sflclean {
namespace {

std::size_t checked_numel(const google::protobuf::RepeatedField<std::int64_t>& shape,
                          std::size_t element_size,
                          std::size_t byte_limit) {
    if (shape.empty() || shape.size() > 8) {
        throw std::invalid_argument("tensor rank must be in [1, 8]");
    }
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0) throw std::invalid_argument("tensor dimensions must be positive");
        const auto value = static_cast<std::size_t>(dim);
        if (value > byte_limit / element_size || count > byte_limit / element_size / value) {
            throw std::invalid_argument("tensor exceeds configured byte limit");
        }
        count *= value;
    }
    return count;
}

void validate_identity(const ::sfl::clean::v1::Tensor& wire,
                       const std::string& expected_name,
                       const std::vector<std::int64_t>& expected_shape) {
    if (wire.name().empty() || wire.name() != expected_name) {
        throw std::invalid_argument("unexpected tensor name: " + wire.name());
    }
    if (wire.shape_size() != static_cast<int>(expected_shape.size())) {
        throw std::invalid_argument("unexpected rank for tensor " + wire.name());
    }
    for (int i = 0; i < wire.shape_size(); ++i) {
        if (wire.shape(i) != expected_shape[static_cast<std::size_t>(i)]) {
            throw std::invalid_argument("unexpected shape for tensor " + wire.name());
        }
    }
}

void append_u32_le(std::string& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

void append_u64_le(std::string& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

std::uint32_t read_u32_le(const char* bytes) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (8U * i);
    }
    return value;
}

std::uint64_t read_u64_le(const char* bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8U * i);
    }
    return value;
}

}  // namespace

void validate_finite_scalar(float value, const char* field_name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(field_name) + " must be finite");
    }
}

::sfl::clean::v1::Tensor encode_float32_tensor(const std::string& name,
                                                const ops::TensorPtr& tensor,
                                                std::size_t byte_limit) {
    if (name.empty() || !tensor || tensor->dtype() != ops::DType::kFloat32) {
        throw std::invalid_argument("float32 tensor encoding requires a name and float32 tensor");
    }
    ::sfl::clean::v1::Tensor wire;
    wire.set_name(name);
    for (const auto dim : tensor->shape()) wire.add_shape(dim);
    wire.set_dtype(::sfl::clean::v1::Tensor::FLOAT32);
    const auto count = checked_numel(wire.shape(), sizeof(float), byte_limit);
    if (count != static_cast<std::size_t>(tensor->numel())) {
        throw std::invalid_argument("tensor shape/element count mismatch");
    }
    std::string bytes;
    bytes.reserve(count * sizeof(float));
    const float* data = tensor->data<float>();
    for (std::size_t i = 0; i < count; ++i) {
        validate_finite_scalar(data[i], name.c_str());
        std::uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        append_u32_le(bytes, bits);
    }
    wire.set_data(std::move(bytes));
    return wire;
}

::sfl::clean::v1::Tensor encode_int64_tensor(const std::string& name,
                                              const std::vector<std::int64_t>& values,
                                              const std::vector<std::int64_t>& shape,
                                              std::size_t byte_limit) {
    if (name.empty()) throw std::invalid_argument("int64 tensor name must not be empty");
    ::sfl::clean::v1::Tensor wire;
    wire.set_name(name);
    for (const auto dim : shape) wire.add_shape(dim);
    wire.set_dtype(::sfl::clean::v1::Tensor::INT64);
    const auto count = checked_numel(wire.shape(), sizeof(std::int64_t), byte_limit);
    if (count != values.size()) throw std::invalid_argument("int64 shape/element count mismatch");
    std::string bytes;
    bytes.reserve(values.size() * sizeof(std::int64_t));
    for (const auto value : values) append_u64_le(bytes, static_cast<std::uint64_t>(value));
    wire.set_data(std::move(bytes));
    return wire;
}

ops::TensorPtr decode_float32_tensor(const ::sfl::clean::v1::Tensor& wire,
                                      const std::string& expected_name,
                                      const std::vector<std::int64_t>& expected_shape,
                                      std::size_t byte_limit) {
    validate_identity(wire, expected_name, expected_shape);
    if (wire.dtype() != ::sfl::clean::v1::Tensor::FLOAT32) {
        throw std::invalid_argument("unexpected dtype for tensor " + wire.name());
    }
    const auto count = checked_numel(wire.shape(), sizeof(float), byte_limit);
    if (wire.data().size() != count * sizeof(float)) {
        throw std::invalid_argument("unexpected byte length for tensor " + wire.name());
    }
    auto result = std::make_shared<ops::Tensor>(expected_shape, ops::DType::kFloat32, ops::kCPU);
    float* output = result->data<float>();
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = read_u32_le(wire.data().data() + i * sizeof(float));
        std::memcpy(output + i, &bits, sizeof(bits));
        validate_finite_scalar(output[i], wire.name().c_str());
    }
    return result;
}

std::vector<std::int64_t> decode_int64_tensor(const ::sfl::clean::v1::Tensor& wire,
                                               const std::string& expected_name,
                                               const std::vector<std::int64_t>& expected_shape,
                                               std::size_t byte_limit) {
    validate_identity(wire, expected_name, expected_shape);
    if (wire.dtype() != ::sfl::clean::v1::Tensor::INT64) {
        throw std::invalid_argument("unexpected dtype for tensor " + wire.name());
    }
    const auto count = checked_numel(wire.shape(), sizeof(std::int64_t), byte_limit);
    if (wire.data().size() != count * sizeof(std::int64_t)) {
        throw std::invalid_argument("unexpected byte length for tensor " + wire.name());
    }
    std::vector<std::int64_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<std::int64_t>(read_u64_le(
            wire.data().data() + i * sizeof(std::int64_t)));
    }
    return values;
}

}  // namespace sflclean
