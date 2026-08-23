cmake_minimum_required(VERSION 3.24)
project(sfl_clean_android_client VERSION 0.1.0 LANGUAGES C CXX)

if(NOT ANDROID)
    message(WARNING "The production client is intended for Android; host builds are for tests only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MFT_SOURCE_DIR "" CACHE PATH "Patched pristine MobileFineTuner source root")
set(SFL_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../proto" CACHE PATH "Protocol source directory")
set(SFL_HOST_PROTOC "" CACHE FILEPATH "Host protoc executable")
set(SFL_HOST_GRPC_CPP_PLUGIN "" CACHE FILEPATH "Host grpc_cpp_plugin executable")
option(SFL_BUILD_TESTS "Build Android-client unit tests" OFF)

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
    src/client_options.cpp
    src/client_runner.cpp
    src/lora_state.cpp
    src/metrics.cpp
    src/tensor_codec.cpp
    src/wiki_text_stream.cpp)
target_include_directories(sfl_clean_client_core PUBLIC include)
target_compile_options(sfl_clean_client_core PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Werror>)
target_link_libraries(sfl_clean_client_core
    PUBLIC
        MobileFineTuner::operators
        sfl_clean_protocol
        gRPC::grpc++)

add_executable(sfl_android_client src/main.cpp)
target_link_libraries(sfl_android_client PRIVATE sfl_clean_client_core)

if(ANDROID)
    target_link_libraries(sfl_android_client PRIVATE log android)
    add_custom_command(TARGET sfl_android_client POST_BUILD
        COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:sfl_android_client>"
        COMMENT "Stripping Android client for ADB staging"
        VERBATIM)
endif()

install(TARGETS sfl_android_client RUNTIME DESTINATION bin)

if(SFL_BUILD_TESTS)
    enable_testing()
    add_executable(test_wikitext_stream
        tests/test_wikitext_stream.cpp
        src/wiki_text_stream.cpp)
    target_include_directories(test_wikitext_stream PRIVATE include)
    add_test(NAME WikiTextStream COMMAND test_wikitext_stream)

    add_executable(test_tensor_codec tests/test_tensor_codec.cpp)
    target_link_libraries(test_tensor_codec PRIVATE sfl_clean_client_core)
    add_test(NAME TensorCodec COMMAND test_tensor_codec)
endif()
