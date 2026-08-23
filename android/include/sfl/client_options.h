#pragma once

#include <cstdint>
#include <string>

namespace sflclean {

struct ClientOptions {
    std::string client_id;
    std::string run_id;
    std::string model_dir;
    std::string dataset_path;
    std::string suffix_target;
    std::string coordinator_target;
    std::string metrics_path;
    std::string config_path;

    std::uint32_t protocol_version = 1;
    std::uint32_t cut_layer = 1;
    std::uint32_t local_steps = 1;
    std::uint32_t batch_size = 1;
    std::uint32_t sequence_length = 16;
    std::uint32_t client_index = 0;
    std::uint32_t client_count = 1;
    std::uint32_t lora_rank = 8;
    float lora_alpha = 32.0F;
    float learning_rate = 2.0e-4F;
    float max_grad_norm = 1.0F;
    std::uint32_t rpc_deadline_ms = 30000;
    std::uint32_t poll_interval_ms = 1000;
    std::uint32_t max_message_bytes = 64U * 1024U * 1024U;
};

ClientOptions parse_client_options(int argc, char** argv);
std::string client_usage(const char* program);

}  // namespace sflclean
