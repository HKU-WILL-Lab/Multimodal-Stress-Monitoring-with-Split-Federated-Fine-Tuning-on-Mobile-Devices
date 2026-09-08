#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sflclean {

struct InferenceResult {
    int valence = 0;
    int arousal = 0;
    std::string assessment;
    std::string generated_text;
};

class LocalInference {
public:
    explicit LocalInference(const std::string& deployment_config);
    ~LocalInference();
    LocalInference(const LocalInference&) = delete;
    LocalInference& operator=(const LocalInference&) = delete;

    InferenceResult infer(const std::vector<std::uint8_t>& sensor_window);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sflclean
