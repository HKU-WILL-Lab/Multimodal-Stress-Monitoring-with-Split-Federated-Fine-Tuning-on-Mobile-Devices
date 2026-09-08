#pragma once

#include "sfl/client_options.h"

#include <cstdint>

namespace sflclean {

class TrainingObserver {
public:
    virtual ~TrainingObserver() = default;
    virtual bool should_cancel() const = 0;
    virtual void on_phase(const char* phase, const char* message) = 0;
    virtual void on_step(std::uint64_t round,
                         std::uint32_t local_step,
                         std::uint32_t total_local_steps,
                         float loss) = 0;
};

int run_client(const ClientOptions& options);
int run_client(const ClientOptions& options, TrainingObserver* observer);

}  // namespace sflclean
