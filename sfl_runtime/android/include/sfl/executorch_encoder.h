#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sflclean {

// Frozen mobile encoder backed by an exported ExecuTorch program.  Shapes are
// supplied by the deployment manifest rather than hard-coded to one encoder.
class ExecuTorchEncoder {
public:
    ExecuTorchEncoder(std::string program_path,
                      std::size_t input_channels,
                      std::size_t input_length,
                      std::size_t output_tokens,
                      std::size_t output_width);
    ~ExecuTorchEncoder();

    ExecuTorchEncoder(const ExecuTorchEncoder&) = delete;
    ExecuTorchEncoder& operator=(const ExecuTorchEncoder&) = delete;

    std::vector<float> encode(const std::vector<float>& channel_major_window);
    std::size_t input_size() const { return input_channels_ * input_length_; }
    std::size_t output_width() const { return output_width_; }
    std::size_t output_tokens() const { return output_tokens_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t input_channels_;
    std::size_t input_length_;
    std::size_t output_tokens_;
    std::size_t output_width_;
};

}  // namespace sflclean
