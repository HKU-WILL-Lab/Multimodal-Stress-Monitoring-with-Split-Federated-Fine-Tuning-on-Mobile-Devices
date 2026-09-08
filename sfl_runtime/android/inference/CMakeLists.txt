cmake_minimum_required(VERSION 3.24)
project(sfl_local_inference VERSION 0.1.0 LANGUAGES C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "The local inference runtime is built for Android")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MFT_SOURCE_DIR "" CACHE PATH "Pristine MobileFineTuner source root")
set(EXECUTORCH_SOURCE_DIR "" CACHE PATH "ExecuTorch source root")
foreach(required MFT_SOURCE_DIR EXECUTORCH_SOURCE_DIR)
    if("${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} must be provided")
    endif()
endforeach()

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(USE_BLAS OFF CACHE BOOL "" FORCE)
add_subdirectory("${MFT_SOURCE_DIR}/operator" "${CMAKE_BINARY_DIR}/mobile_finetuner")

set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_LLM ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON CACHE BOOL "" FORCE)
add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")

add_library(sfl_local_inference_core STATIC
    ../src/alignment_projector.cpp
    ../src/executorch_encoder.cpp
    ../src/local_inference.cpp)
target_include_directories(sfl_local_inference_core
    PUBLIC ../include
    PRIVATE "${EXECUTORCH_SOURCE_DIR}/third-party/json/include")
target_compile_definitions(sfl_local_inference_core PUBLIC SFL_HAS_EXECUTORCH_ENCODER=1)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(sfl_local_inference_core PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-unused-parameter
        -Wno-sign-compare)
endif()
target_link_libraries(sfl_local_inference_core PUBLIC
    MobileFineTuner::operators
    executorch executorch::backends executorch::kernels
    extension_module extension_tensor
    extension_llm_runner tokenizers::tokenizers)

add_library(wellbeing_inference SHARED ../src/local_inference_jni.cpp)
target_link_libraries(wellbeing_inference PRIVATE sfl_local_inference_core log android)
add_custom_command(TARGET wellbeing_inference POST_BUILD
    COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:wellbeing_inference>"
    COMMENT "Stripping local inference JNI runtime"
    VERBATIM)
