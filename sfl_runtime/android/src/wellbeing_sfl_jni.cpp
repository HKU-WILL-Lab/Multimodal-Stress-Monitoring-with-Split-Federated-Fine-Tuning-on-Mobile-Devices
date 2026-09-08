#include "sfl/client_options.h"
#include "sfl/client_runner.h"

#include <jni.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::string from_java(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* raw = env->GetStringUTFChars(value, nullptr);
    if (raw == nullptr) throw std::runtime_error("unable to read Java string");
    std::string result(raw);
    env->ReleaseStringUTFChars(value, raw);
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

struct Snapshot {
    std::string phase = "IDLE";
    std::string message = "Ready";
    std::uint64_t round = 0;
    std::uint32_t local_step = 0;
    std::uint32_t total_local_steps = 0;
    std::uint32_t total_rounds = 0;
    float loss = 0.0F;
    bool has_loss = false;
};

class Session final : public sflclean::TrainingObserver {
public:
    ~Session() override { close(); }

    bool start(sflclean::ClientOptions options) {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (worker_.joinable()) {
            if (running_.load()) return false;
            worker_.join();
        }
        cancel_.store(false);
        running_.store(true);
        {
            std::lock_guard<std::mutex> snapshot_lock(snapshot_mutex_);
            snapshot_ = Snapshot{};
            snapshot_.phase = "STARTING";
            snapshot_.message = "Loading model and training assets";
            snapshot_.total_rounds = options.total_rounds;
            snapshot_.total_local_steps = options.local_steps;
        }
        worker_ = std::thread([this, options = std::move(options)]() {
            try {
                const int result = sflclean::run_client(options, this);
                if (result != 0) update_phase("FAILED", "Native training client exited with an error");
            } catch (const std::exception& error) {
                update_phase("FAILED", error.what());
            } catch (...) {
                update_phase("FAILED", "Unknown native training error");
            }
            running_.store(false);
        });
        return true;
    }

    bool should_cancel() const override { return cancel_.load(); }

    void on_phase(const char* phase, const char* message) override {
        update_phase(phase == nullptr ? "FAILED" : phase,
                     message == nullptr ? "" : message);
    }

    void on_step(std::uint64_t round, std::uint32_t local_step,
                 std::uint32_t total_local_steps, float loss) override {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        snapshot_.phase = "TRAINING";
        snapshot_.message = "Training on labeled sensor windows";
        snapshot_.round = round;
        snapshot_.local_step = local_step;
        snapshot_.total_local_steps = total_local_steps;
        snapshot_.loss = loss;
        snapshot_.has_loss = true;
    }

    void cancel() {
        if (!running_.load()) return;
        cancel_.store(true);
        update_phase("CANCELLING", "Stopping after the current RPC or training step");
    }

    void close() {
        cancel();
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (worker_.joinable()) worker_.join();
    }

    std::string status_json() const {
        Snapshot value;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            value = snapshot_;
        }
        std::ostringstream out;
        out << "{\"phase\":\"" << json_escape(value.phase)
            << "\",\"round\":" << value.round
            << ",\"totalRounds\":" << value.total_rounds
            << ",\"localStep\":" << value.local_step
            << ",\"totalLocalSteps\":" << value.total_local_steps
            << ",\"loss\":";
        if (value.has_loss) out << value.loss; else out << "null";
        out << ",\"message\":\"" << json_escape(value.message) << "\"}";
        return out.str();
    }

private:
    void update_phase(std::string phase, std::string message) {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        snapshot_.phase = std::move(phase);
        snapshot_.message = std::move(message);
    }

    std::atomic<bool> cancel_{false};
    std::atomic<bool> running_{false};
    mutable std::mutex snapshot_mutex_;
    std::mutex worker_mutex_;
    Snapshot snapshot_;
    std::thread worker_;
};

Session& session() {
    static Session value;
    return value;
}

sflclean::ClientOptions make_options(JNIEnv* env, jstring client_id, jstring config,
                                     jstring model_dir, jstring encoder_program,
                                     jstring dataset, jstring suffix_target,
                                     jstring coordinator_target,
                                     jint encoder_input_length) {
    std::vector<std::string> values = {
        "wellbeing_sfl", "--client-id", from_java(env, client_id),
        "--config", from_java(env, config), "--model-dir", from_java(env, model_dir),
        "--encoder-pte", from_java(env, encoder_program),
        "--dataset", from_java(env, dataset),
        "--suffix", from_java(env, suffix_target),
        "--coordinator", from_java(env, coordinator_target),
        "--encoder-input-length", std::to_string(encoder_input_length),
    };
    std::vector<char*> argv;
    argv.reserve(values.size());
    for (auto& value : values) argv.push_back(value.data());
    return sflclean::parse_client_options(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_org_mobihoc_wellbeing_phone_training_NativeTrainingBridge_nativeAvailable(
    JNIEnv*, jclass) {
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_mobihoc_wellbeing_phone_training_NativeTrainingBridge_start(
    JNIEnv* env, jclass, jstring client_id, jstring config, jstring model_dir,
    jstring encoder_program, jstring dataset, jstring suffix_target,
    jstring coordinator_target, jint encoder_input_length) {
    try {
        return session().start(make_options(
                   env, client_id, config, model_dir, encoder_program, dataset,
                   suffix_target, coordinator_target, encoder_input_length))
                   ? JNI_TRUE
                   : JNI_FALSE;
    } catch (const std::exception& error) {
        jclass exception = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exception, error.what());
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_mobihoc_wellbeing_phone_training_NativeTrainingBridge_status(JNIEnv* env, jclass) {
    const std::string value = session().status_json();
    return env->NewStringUTF(value.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_org_mobihoc_wellbeing_phone_training_NativeTrainingBridge_cancel(JNIEnv*, jclass) {
    session().cancel();
}

extern "C" JNIEXPORT void JNICALL
Java_org_mobihoc_wellbeing_phone_training_NativeTrainingBridge_close(JNIEnv*, jclass) {
    session().close();
}
