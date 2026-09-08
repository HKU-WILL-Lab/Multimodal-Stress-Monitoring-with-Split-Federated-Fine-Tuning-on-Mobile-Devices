cmake_minimum_required(VERSION 3.24)
project(sfl_clean_android_client VERSION 0.1.0 LANGUAGES C CXX)

if(NOT ANDROID)
    message(WARNING "The production client is intended for Android; host builds are for tests only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MFT_SOURCE_DIR "" CACHE PATH "Pristine MobileFineTuner source root")
set(SFL_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../proto" CACHE PATH "Protocol source directory")
set(SFL_HOST_PROTOC "" CACHE FILEPATH "Host protoc executable")
set(SFL_HOST_GRPC_CPP_PLUGIN "" CACHE FILEPATH "Host grpc_cpp_plugin executable")
option(SFL_BUILD_TESTS "Build Android-client unit tests" OFF)
option(SFL_ENABLE_EXECUTORCH_ENCODER "Build the frozen ExecuTorch sensor encoder" OFF)
set(EXECUTORCH_SOURCE_DIR "" CACHE PATH "ExecuTorch source root used to export the encoder")

foreach(required MFT_SOURCE_DIR SFL_PROTO_DIR SFL_HOST_PROTOC SFL_HOST_GRPC_CPP_PLUGIN)
    if("${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} must be provided")
    endif()
endforeach()

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(USE_BLAS OFF CACHE BOOL "" FORCE)
add_subdirectory("${MFT_SOURCE_DIR}/operator" "${CMAKE_BINARY_DIR}/mobile_finetuner")

set(PROTO_FILE "${SFL_PROTO_DIR}/sfl_clean.proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PROTO_SOURCE "${GENERATED_DIR}/sfl_clean.pb.cc")
set(PROTO_HEADER "${GENERATED_DIR}/sfl_clean.pb.h")
set(GRPC_SOURCE "${GENERATED_DIR}/sfl_clean.grpc.pb.cc")
set(GRPC_HEADER "${GENERATED_DIR}/sfl_clean.grpc.pb.h")

add_custom_command(
    OUTPUT "${PROTO_SOURCE}" "${PROTO_HEADER}" "${GRPC_SOURCE}" "${GRPC_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${GENERATED_DIR}"
    COMMAND "${SFL_HOST_PROTOC}"
        "--proto_path=${SFL_PROTO_DIR}"
        "--cpp_out=${GENERATED_DIR}"
        "--grpc_out=${GENERATED_DIR}"
        "--plugin=protoc-gen-grpc=${SFL_HOST_GRPC_CPP_PLUGIN}"
        "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}" "${SFL_HOST_PROTOC}" "${SFL_HOST_GRPC_CPP_PLUGIN}"
    VERBATIM)

add_library(sfl_clean_protocol STATIC
    "${PROTO_SOURCE}"
    "${GRPC_SOURCE}")
target_include_directories(sfl_clean_protocol PUBLIC "${GENERATED_DIR}")
target_link_libraries(sfl_clean_protocol PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_library(sfl_clean_client_core STATIC
    src/alignment_projector.cpp
    src/client_options.cpp
    src/client_runner.cpp
    src/llama_prefix_block.cpp
    src/lora_state.cpp
    src/metrics.cpp
    src/sensor_text_stream.cpp
    src/tensor_codec.cpp)
target_include_directories(sfl_clean_client_core PUBLIC include)
target_compile_definitions(sfl_clean_client_core PUBLIC USE_NEW_AUTOGRAD_ENGINE=1)
target_compile_options(sfl_clean_client_core PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Werror>)
target_link_libraries(sfl_clean_client_core
    PUBLIC
        MobileFineTuner::operators
        sfl_clean_protocol
        gRPC::grpc++)

if(SFL_ENABLE_EXECUTORCH_ENCODER)
    if(EXECUTORCH_SOURCE_DIR STREQUAL "")
        message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR is required when SFL_ENABLE_EXECUTORCH_ENCODER=ON")
    endif()
    set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_LLM OFF CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER OFF CACHE BOOL "" FORCE)
    add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")
    target_sources(sfl_clean_client_core PRIVATE
        src/executorch_encoder.cpp)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set_source_files_properties(src/executorch_encoder.cpp PROPERTIES
            COMPILE_OPTIONS
                "-Wno-error=gnu-zero-variadic-macro-arguments;-Wno-error=unused-parameter;-Wno-error=sign-compare")
    endif()
    target_link_libraries(sfl_clean_client_core PUBLIC
        executorch executorch::backends executorch::kernels
        extension_module extension_tensor)
    target_compile_definitions(sfl_clean_client_core PUBLIC SFL_HAS_EXECUTORCH_ENCODER=1)

endif()

add_executable(sfl_android_client src/main.cpp)
target_link_libraries(sfl_android_client PRIVATE sfl_clean_client_core)

if(ANDROID)
    add_library(wellbeing_sfl SHARED src/wellbeing_sfl_jni.cpp)
    target_link_libraries(wellbeing_sfl PRIVATE sfl_clean_client_core log android)
    add_custom_command(TARGET wellbeing_sfl POST_BUILD
        COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:wellbeing_sfl>"
        COMMENT "Stripping app JNI runtime"
        VERBATIM)

    target_link_libraries(sfl_android_client PRIVATE log android)
    add_custom_command(TARGET sfl_android_client POST_BUILD
        COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:sfl_android_client>"
        COMMENT "Stripping Android client for ADB staging"
        VERBATIM)
endif()

install(TARGETS sfl_android_client RUNTIME DESTINATION bin)

if(SFL_BUILD_TESTS)
    enable_testing()
    add_executable(test_tensor_codec tests/test_tensor_codec.cpp)
    target_link_libraries(test_tensor_codec PRIVATE sfl_clean_client_core)
    add_test(NAME TensorCodec COMMAND test_tensor_codec)

    add_executable(test_sensor_text_stream
        tests/test_sensor_text_stream.cpp
        src/sensor_text_stream.cpp)
    target_include_directories(test_sensor_text_stream PRIVATE include)
    target_link_libraries(test_sensor_text_stream PRIVATE protobuf::libprotobuf)
    add_test(NAME SensorTextStream COMMAND test_sensor_text_stream)
endif()
