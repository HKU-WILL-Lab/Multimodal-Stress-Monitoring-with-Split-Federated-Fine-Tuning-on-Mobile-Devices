#include "sfl/local_inference.h"

#include <jni.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::mutex& engine_mutex() {
    static std::mutex value;
    return value;
}

std::unique_ptr<sflclean::LocalInference>& engine() {
    static std::unique_ptr<sflclean::LocalInference> value;
    return value;
}

void throw_java(JNIEnv* env, const char* type, const std::string& message) {
    jclass exception = env->FindClass(type);
    if (exception != nullptr) env->ThrowNew(exception, message.c_str());
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_org_mobihoc_wellbeing_phone_inference_NativeSflBridge_nativeInferenceAvailable(
    JNIEnv*, jclass) {
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_org_mobihoc_wellbeing_phone_inference_NativeSflBridge_configure(
    JNIEnv* env, jclass, jstring config_path) {
    try {
        auto configured = std::make_unique<sflclean::LocalInference>(from_java(env, config_path));
        std::lock_guard<std::mutex> lock(engine_mutex());
        engine() = std::move(configured);
    } catch (const std::exception& error) {
        throw_java(env, "java/lang/IllegalArgumentException", error.what());
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_mobihoc_wellbeing_phone_inference_NativeSflBridge_configured(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(engine_mutex());
    return engine() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_mobihoc_wellbeing_phone_inference_NativeSflBridge_infer(
    JNIEnv* env, jclass, jbyteArray encoded_window) {
    try {
        if (encoded_window == nullptr) throw std::invalid_argument("sensor window is null");
        const jsize length = env->GetArrayLength(encoded_window);
        if (length <= 0 || length > 16 * 1024 * 1024) {
            throw std::invalid_argument("sensor window byte length is invalid");
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
        env->GetByteArrayRegion(encoded_window, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
        if (env->ExceptionCheck()) return nullptr;
        std::lock_guard<std::mutex> lock(engine_mutex());
        if (!engine()) throw std::runtime_error("local inference is not configured");
        const auto result = engine()->infer(bytes);
        std::ostringstream json;
        json << "{\"valence\":" << result.valence
             << ",\"arousal\":" << result.arousal
             << ",\"assessment\":\"" << json_escape(result.assessment)
             << "\",\"generatedText\":\"" << json_escape(result.generated_text)
             << "\"}";
        return env->NewStringUTF(json.str().c_str());
    } catch (const std::exception& error) {
        throw_java(env, "java/lang/RuntimeException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_mobihoc_wellbeing_phone_inference_NativeSflBridge_close(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(engine_mutex());
    engine().reset();
}
