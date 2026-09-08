#include "finetune_ops/core/tensor.h"
#include "sfl/tensor_codec.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    const std::vector<float> values{1.25F, -2.5F, 3.0F, 4.5F};
    auto tensor = std::make_shared<ops::Tensor>(
        std::vector<std::int64_t>{2, 2}, values.data(), ops::DType::kFloat32, ops::kCPU);
    auto wire = sflclean::encode_float32_tensor("stable", tensor, 1024);
    auto decoded = sflclean::decode_float32_tensor(wire, "stable", {2, 2}, 1024);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (decoded->data<float>()[index] != values[index]) {
            throw std::runtime_error("float32 little-endian round trip failed");
        }
    }

    auto integers = sflclean::encode_int64_tensor("ids", {-1, 0, 4}, {1, 3}, 1024);
    if (sflclean::decode_int64_tensor(integers, "ids", {1, 3}, 1024) !=
        std::vector<std::int64_t>({-1, 0, 4})) {
        throw std::runtime_error("int64 little-endian round trip failed");
    }
    wire.set_data(std::string(3, '\0'));
    try {
        (void)sflclean::decode_float32_tensor(wire, "stable", {2, 2}, 1024);
        throw std::runtime_error("malformed byte length was accepted");
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
