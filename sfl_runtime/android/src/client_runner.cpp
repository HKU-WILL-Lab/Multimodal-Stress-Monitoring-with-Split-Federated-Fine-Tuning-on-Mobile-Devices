#include "sfl/client_runner.h"

#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/utils.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"
#include "sfl/alignment_projector.h"
#include "sfl/client_options.h"
#include "sfl/lora_state.h"
#include "sfl/llama_prefix_block.h"
#include "sfl/metrics.h"
#include "sfl/sensor_text_stream.h"
#include "sfl/tensor_codec.h"
#include "sfl_clean.grpc.pb.h"

#ifdef SFL_HAS_EXECUTORCH_ENCODER
#include "sfl/executorch_encoder.h"
#endif

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sflclean {
namespace {

using Protocol = ::sfl::clean::v1::RoundState;

struct RoundView {
    Protocol state = ::sfl::clean::v1::ROUND_STATE_UNSPECIFIED;
    std::uint64_t global_round = 0;
    google::protobuf::RepeatedPtrField<::sfl::clean::v1::Tensor> prefix_tensors;
};

std::chrono::system_clock::time_point deadline(std::uint32_t milliseconds) {
    return std::chrono::system_clock::now() + std::chrono::milliseconds(milliseconds);
}

std::runtime_error rpc_failure(const char* rpc, const grpc::Status& status) {
    return std::runtime_error(
        std::string(rpc) + " failed (gRPC " + std::to_string(status.error_code()) + "): " +
        status.error_message() +
        (status.error_details().empty() ? "" : " [" + status.error_details() + "]"));
}

void validate_round_state(Protocol state, const char* rpc) {
    if (state != ::sfl::clean::v1::TRAIN &&
        state != ::sfl::clean::v1::WAIT &&
        state != ::sfl::clean::v1::DONE) {
        throw std::runtime_error(std::string(rpc) + " returned an unspecified round state");
    }
}

float frozen_weight_value(const ops::TensorPtr& weight, std::int64_t index) {
    if (weight->dtype() == ops::kFloat32) return weight->data<float>()[index];
    if (weight->dtype() == ops::kFloat16) {
        return ops::fp16_bits_to_float32(weight->data<std::uint16_t>()[index]);
    }
    if (weight->dtype() == ops::kBFloat16) {
        return ops::bf16_bits_to_float32(weight->data<std::uint16_t>()[index]);
    }
    throw std::invalid_argument("unsupported Llama embedding dtype");
}

ops::TensorPtr load_llama32_embedding(const std::string& model_dir) {
    ops::SafeTensorsModelReader reader(model_dir);
    reader.parse_headers();
    ops::SafeTensorsLoadOptions load_options;
    load_options.verbose = false;
    load_options.preserve_low_precision_key_substrings = {"embed_tokens"};
    auto loaded = reader.load_tensors_mapped(
        {{"embed_tokens.weight", "model.embed_tokens.weight"}}, load_options);
    const auto found = loaded.find("embed_tokens.weight");
    if (found == loaded.end() || found->second->shape() != std::vector<std::int64_t>{128256, 2048}) {
        throw std::invalid_argument(
            "model assets are not the expected Llama 3.2 1B embedding table");
    }
    return found->second;
}

ops::TensorPtr embed_text(const ops::TensorPtr& embedding,
                          const std::vector<std::int64_t>& token_ids,
                          std::uint32_t batch,
                          std::uint32_t sequence) {
    constexpr std::int64_t width = 2048;
    auto output = ops::zeros({batch, sequence, width});
    float* destination = output->data<float>();
    for (std::size_t position = 0; position < token_ids.size(); ++position) {
        const auto token = token_ids[position];
        if (token < 0 || token >= 128256) throw std::invalid_argument("Llama token id is out of range");
        const std::int64_t start = token * width;
        for (std::int64_t column = 0; column < width; ++column) {
            destination[position * width + column] = frozen_weight_value(embedding, start + column);
        }
    }
    return output;
}

struct ConditionedBatch {
    ops::TensorPtr embeddings;
    std::vector<std::int64_t> token_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> loss_mask;
    std::vector<std::uint32_t> sensor_offsets;
};

class InsertSensorBackward final : public ops::BackwardFunction {
public:
    InsertSensorBackward(std::vector<std::int64_t> sensor_shape,
                         std::vector<std::int64_t> text_shape,
                         std::vector<std::uint32_t> offsets)
        : sensor_shape_(std::move(sensor_shape)),
          text_shape_(std::move(text_shape)),
          offsets_(std::move(offsets)) {}

    std::vector<ops::TensorPtr> apply(
        const ops::TensorPtr& gradient) override {
        const auto batch = sensor_shape_[0];
        const auto sensor_length = sensor_shape_[1];
        const auto text_length = text_shape_[1];
        const auto width = sensor_shape_[2];
        if (!gradient || gradient->shape() != std::vector<std::int64_t>{
                batch, sensor_length + text_length, width}) {
            throw std::invalid_argument("conditioned embedding gradient shape is invalid");
        }
        auto sensor_gradient = ops::zeros(sensor_shape_);
        auto text_gradient = ops::zeros(text_shape_);
        const float* source = gradient->data<float>();
        float* sensor_target = sensor_gradient->data<float>();
        float* text_target = text_gradient->data<float>();
        for (std::int64_t row = 0; row < batch; ++row) {
            const auto prompt = static_cast<std::int64_t>(
                offsets_[static_cast<std::size_t>(row)]);
            const auto source_row = row * (sensor_length + text_length) * width;
            const auto text_row = row * text_length * width;
            const auto sensor_row = row * sensor_length * width;
            std::copy(source + source_row,
                      source + source_row + prompt * width,
                      text_target + text_row);
            std::copy(source + source_row + prompt * width,
                      source + source_row + (prompt + sensor_length) * width,
                      sensor_target + sensor_row);
            std::copy(source + source_row + (prompt + sensor_length) * width,
                      source + source_row + (sensor_length + text_length) * width,
                      text_target + text_row + prompt * width);
        }
        return {sensor_gradient, text_gradient};
    }

private:
    std::vector<std::int64_t> sensor_shape_;
    std::vector<std::int64_t> text_shape_;
    std::vector<std::uint32_t> offsets_;
};

ConditionedBatch insert_sensor_after_prompt(
                                     const ops::TensorPtr& sensor,
                                     const ops::TensorPtr& text,
                                     const SensorTextBatch& batch) {
    if (sensor->shape().size() != 3 || text->shape().size() != 3 ||
        sensor->shape()[0] != text->shape()[0] || sensor->shape()[2] != text->shape()[2]) {
        throw std::invalid_argument("sensor and text embedding shapes cannot be combined");
    }
    const auto sensor_length = sensor->shape()[1];
    const auto text_length = text->shape()[1];
    const auto width = sensor->shape()[2];
    auto output = ops::zeros({sensor->shape()[0], sensor_length + text_length, width});
    float* destination = output->data<float>();
    const float* sensor_data = sensor->data<float>();
    const float* text_data = text->data<float>();
    ConditionedBatch result;
    result.embeddings = output;
    result.token_ids.reserve(static_cast<std::size_t>(batch.batch_size) * (text_length + sensor_length));
    result.attention_mask.reserve(result.token_ids.capacity());
    result.loss_mask.reserve(result.token_ids.capacity());
    result.sensor_offsets.reserve(static_cast<std::size_t>(batch.batch_size));
    for (std::int64_t row = 0; row < sensor->shape()[0]; ++row) {
        const auto text_offset = static_cast<std::size_t>(row) * text_length;
        std::uint32_t prompt_length = static_cast<std::uint32_t>(text_length);
        for (std::uint32_t position = 0; position < text_length; ++position) {
            if (batch.loss_mask[text_offset + position] != 0) {
                prompt_length = position;
                break;
            }
        }
        result.sensor_offsets.push_back(prompt_length);
        const auto output_offset = row * (sensor_length + text_length) * width;
        std::copy(text_data + row * text_length * width,
                  text_data + row * text_length * width + prompt_length * width,
                  destination + output_offset);
        std::copy(sensor_data + row * sensor_length * width,
                  sensor_data + (row + 1) * sensor_length * width,
                  destination + output_offset + prompt_length * width);
        std::copy(text_data + row * text_length * width + prompt_length * width,
                  text_data + (row + 1) * text_length * width,
                  destination + output_offset + (prompt_length + sensor_length) * width);

        result.token_ids.insert(result.token_ids.end(),
            batch.token_ids.begin() + text_offset,
            batch.token_ids.begin() + text_offset + prompt_length);
        result.attention_mask.insert(result.attention_mask.end(),
            batch.attention_mask.begin() + text_offset,
            batch.attention_mask.begin() + text_offset + prompt_length);
        result.loss_mask.insert(result.loss_mask.end(), prompt_length, 0);
        result.token_ids.insert(result.token_ids.end(), sensor_length, 0);
        result.attention_mask.insert(result.attention_mask.end(), sensor_length, 1);
        result.loss_mask.insert(result.loss_mask.end(), sensor_length, 0);
        result.token_ids.insert(result.token_ids.end(),
            batch.token_ids.begin() + text_offset + prompt_length,
            batch.token_ids.begin() + text_offset + text_length);
        result.attention_mask.insert(result.attention_mask.end(),
            batch.attention_mask.begin() + text_offset + prompt_length,
            batch.attention_mask.begin() + text_offset + text_length);
        result.loss_mask.insert(result.loss_mask.end(),
            batch.loss_mask.begin() + text_offset + prompt_length,
            batch.loss_mask.begin() + text_offset + text_length);
    }
    if (sensor->requires_grad() || text->requires_grad()) {
        output->set_requires_grad(true);
        auto backward = std::make_shared<InsertSensorBackward>(
            sensor->shape(), text->shape(), result.sensor_offsets);
#ifdef USE_NEW_AUTOGRAD_ENGINE
        ops::autograd::Engine::instance().register_node(
            output, {sensor, text}, backward);
#else
        output->set_grad_fn(
            [backward](const ops::TensorPtr& gradient) {
                return backward->apply(gradient);
            });
#endif
    }
    return result;
}

class RpcClients {
public:
    explicit RpcClients(const ClientOptions& options) : options_(options) {
        const std::uint64_t bytes64 = options.max_message_bytes;
        if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("max-message-mib exceeds the gRPC integer limit");
        }
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(static_cast<int>(bytes64));
        args.SetMaxSendMessageSize(static_cast<int>(bytes64));
        suffix_ = ::sfl::clean::v1::SuffixService::NewStub(grpc::CreateCustomChannel(
            options.suffix_target, grpc::InsecureChannelCredentials(), args));
        coordinator_ = ::sfl::clean::v1::RoundCoordinator::NewStub(grpc::CreateCustomChannel(
            options.coordinator_target, grpc::InsecureChannelCredentials(), args));
    }

    RoundView fetch_or_bootstrap(
        std::uint64_t last_completed_round,
        const std::vector<::sfl::clean::v1::Tensor>& initial_tensors) {
        ::sfl::clean::v1::FetchRoundRequest request;
        request.set_protocol_version(options_.protocol_version);
        request.set_client_id(options_.client_id);
        request.set_last_completed_round(last_completed_round);
        ::sfl::clean::v1::FetchRoundResponse response;
        grpc::ClientContext context;
        context.set_deadline(deadline(options_.rpc_deadline_ms));
        const grpc::Status status = coordinator_->FetchRound(&context, request, &response);
        if (status.ok()) {
            validate_round_state(response.state(), "FetchRound");
            if (response.state() != ::sfl::clean::v1::TRAIN || !response.prefix_tensors().empty()) {
                RoundView view;
                view.state = response.state();
                view.global_round = response.global_round();
                view.prefix_tensors.CopyFrom(response.prefix_tensors());
                return view;
            }
        } else if (status.error_code() != grpc::StatusCode::FAILED_PRECONDITION &&
                   status.error_code() != grpc::StatusCode::NOT_FOUND) {
            throw rpc_failure("FetchRound", status);
        }

        ::sfl::clean::v1::BootstrapRequest bootstrap;
        bootstrap.set_protocol_version(options_.protocol_version);
        bootstrap.set_client_id(options_.client_id);
        for (const auto& tensor : initial_tensors) *bootstrap.add_prefix_tensors() = tensor;
        ::sfl::clean::v1::BootstrapResponse bootstrapped;
        grpc::ClientContext bootstrap_context;
        bootstrap_context.set_deadline(deadline(options_.rpc_deadline_ms));
        const grpc::Status bootstrap_status = coordinator_->Bootstrap(
            &bootstrap_context, bootstrap, &bootstrapped);
        if (!bootstrap_status.ok()) throw rpc_failure("Bootstrap", bootstrap_status);
        validate_round_state(bootstrapped.state(), "Bootstrap");
        RoundView view;
        view.state = bootstrapped.state();
        view.global_round = bootstrapped.global_round();
        view.prefix_tensors.CopyFrom(bootstrapped.prefix_tensors());
        return view;
    }

    ::sfl::clean::v1::SplitStepResponse train_split_step(
        const ::sfl::clean::v1::SplitStepRequest& request) {
        ::sfl::clean::v1::SplitStepResponse response;
        grpc::ClientContext context;
        context.set_deadline(deadline(options_.rpc_deadline_ms));
        const grpc::Status status = suffix_->TrainSplitStep(&context, request, &response);
        if (!status.ok()) throw rpc_failure("TrainSplitStep", status);
        return response;
    }

    ::sfl::clean::v1::SubmitUpdateResponse submit(
        const ::sfl::clean::v1::SubmitUpdateRequest& request) {
        ::sfl::clean::v1::SubmitUpdateResponse response;
        grpc::ClientContext context;
        context.set_deadline(deadline(options_.rpc_deadline_ms));
        const grpc::Status status = coordinator_->SubmitUpdate(&context, request, &response);
        if (!status.ok()) throw rpc_failure("SubmitUpdate", status);
        validate_round_state(response.state(), "SubmitUpdate");
        if (response.global_round() < request.global_round()) {
            throw std::runtime_error("SubmitUpdate returned a regressing global round");
        }
        return response;
    }

private:
    const ClientOptions& options_;
    std::unique_ptr<::sfl::clean::v1::SuffixService::Stub> suffix_;
    std::unique_ptr<::sfl::clean::v1::RoundCoordinator::Stub> coordinator_;
};

}  // namespace

int run_client(const ClientOptions& options) {
    return run_client(options, nullptr);
}

int run_client(const ClientOptions& options, TrainingObserver* observer) {
#ifndef SFL_HAS_EXECUTORCH_ENCODER
    (void)observer;
    throw std::runtime_error(
        "the client was built without SFL_ENABLE_EXECUTORCH_ENCODER");
#else
    if (observer) observer->on_phase("STARTING", "Loading frozen encoder and Llama embedding");
    ExecuTorchEncoder encoder(
        options.encoder_program, options.encoder_input_channels,
        options.encoder_input_length, options.sensor_tokens,
        options.encoder_output_width);
    auto embedding = load_llama32_embedding(options.model_dir);
    AlignmentProjector projector(options.encoder_output_width, 2048);
    std::vector<std::unique_ptr<LlamaPrefixBlock>> prefix_blocks;
    auto named_parameters = projector.named_parameters();
    for (std::uint32_t layer = 0; layer < options.cut_layer; ++layer) {
        auto block = std::make_unique<LlamaPrefixBlock>(
            options.model_dir, options.lora_rank, options.lora_alpha, 42, layer);
        auto block_parameters = block->named_parameters();
        named_parameters.insert(named_parameters.end(), block_parameters.begin(), block_parameters.end());
        prefix_blocks.push_back(std::move(block));
    }
    std::vector<ops::TensorPtr> parameters;
    parameters.reserve(named_parameters.size());
    for (const auto& item : named_parameters) parameters.push_back(item.tensor);

    ops::AdamConfig adam_config(options.learning_rate, 0.9F, 0.999F, 1.0e-8F,
                                0.0F, options.max_grad_norm, false);
    ops::Adam optimizer(adam_config);

    SensorTextStream stream(
        options.dataset_path, options.client_index, options.client_count,
        options.encoder_input_channels * options.encoder_input_length,
        options.sequence_length);

    const std::size_t tensor_limit =
        static_cast<std::size_t>(options.max_message_bytes);
    auto initial_state = encode_prefix_lora(named_parameters, tensor_limit);
    RpcClients rpc(options);
    JsonlMetricWriter metrics(options.metrics_path);

    bool completed_any_round = false;
    std::uint64_t last_completed_round = 0;
    std::uint64_t last_server_step = 0;
    while (true) {
        if (observer && observer->should_cancel()) {
            observer->on_phase("IDLE", "Stopped by user");
            return 0;
        }
        RoundView round = rpc.fetch_or_bootstrap(last_completed_round, initial_state);
        if (round.state == ::sfl::clean::v1::DONE) break;
        if (round.state == ::sfl::clean::v1::WAIT) {
            if (observer) observer->on_phase("WAITING", "Waiting for other clients");
            std::this_thread::sleep_for(std::chrono::milliseconds(options.poll_interval_ms));
            continue;
        }
        if (completed_any_round && round.global_round <= last_completed_round) {
            throw std::runtime_error("coordinator returned TRAIN for an already completed round");
        }
        load_prefix_lora(round.prefix_tensors, named_parameters, tensor_limit);

        std::uint64_t processed_sequences = 0;
        for (std::uint32_t local_step = 0; local_step < options.local_steps; ++local_step) {
            if (observer && observer->should_cancel()) {
                observer->on_phase("IDLE", "Stopped by user");
                return 0;
            }
            SensorTextBatch batch = stream.next_batch(options.batch_size);
            if (batch.batch_size == 0) {
                throw std::runtime_error("assigned dataset partition was exhausted before local training finished");
            }
            processed_sequences += batch.processed_sequences;
            std::vector<float> encoded;
            encoded.reserve(static_cast<std::size_t>(batch.batch_size) *
                            options.sensor_tokens * options.encoder_output_width);
            const std::size_t window_size = batch.sensor_values_per_example;
            for (std::uint32_t row = 0; row < batch.batch_size; ++row) {
                std::vector<float> window(
                    batch.sensor_values.begin() + row * window_size,
                    batch.sensor_values.begin() + (row + 1) * window_size);
                auto features = encoder.encode(window);
                encoded.insert(encoded.end(), features.begin(), features.end());
            }
            auto encoder_features = std::make_shared<ops::Tensor>(
                std::vector<std::int64_t>{batch.batch_size, options.sensor_tokens,
                                          options.encoder_output_width},
                encoded.data(), ops::kFloat32, ops::kCPU);
            auto projected_sensor = projector.forward(encoder_features);
            auto text_embeddings = embed_text(
                embedding, batch.token_ids, batch.batch_size, batch.sequence_length);
            const std::uint32_t conditioned_length = batch.sequence_length + options.sensor_tokens;
            auto conditioned = insert_sensor_after_prompt(projected_sensor, text_embeddings, batch);
            std::vector<std::int32_t> attention32(
                conditioned.attention_mask.begin(), conditioned.attention_mask.end());
            auto native_attention = std::make_shared<ops::Tensor>(
                std::vector<std::int64_t>{batch.batch_size, conditioned_length},
                attention32.data(), ops::kInt32, ops::kCPU);
            auto activation = conditioned.embeddings;
            for (const auto& block : prefix_blocks) activation = block->forward(activation, native_attention);

            ::sfl::clean::v1::SplitStepRequest request;
            request.set_protocol_version(options.protocol_version);
            request.set_client_id(options.client_id);
            request.set_global_round(round.global_round);
            request.set_local_step(local_step);
            request.set_cut_layer(options.cut_layer);
            *request.mutable_activation() = encode_float32_tensor(
                "boundary_activation", activation, tensor_limit);
            *request.mutable_token_ids() = encode_int64_tensor(
                "token_ids", conditioned.token_ids,
                {batch.batch_size, conditioned_length}, tensor_limit);
            *request.mutable_attention_mask() = encode_int64_tensor(
                "attention_mask", conditioned.attention_mask,
                {batch.batch_size, conditioned_length}, tensor_limit);
            *request.mutable_loss_mask() = encode_int64_tensor(
                "loss_mask", conditioned.loss_mask,
                {batch.batch_size, conditioned_length}, tensor_limit);

            const auto started = std::chrono::steady_clock::now();
            auto response = rpc.train_split_step(request);
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);
            validate_finite_scalar(response.loss(), "loss");
            validate_finite_scalar(response.token_accuracy(), "token_accuracy");
            if (response.token_accuracy() < 0.0F || response.token_accuracy() > 1.0F) {
                throw std::runtime_error("suffix returned token accuracy outside [0, 1]");
            }
            if (response.server_step() <= last_server_step && last_server_step != 0) {
                throw std::runtime_error("suffix server_step did not increase");
            }
            last_server_step = response.server_step();

            auto activation_gradient = decode_float32_tensor(
                response.activation_gradient(), "boundary_activation_gradient",
                activation->shape(), tensor_limit);
            activation->backward(activation_gradient);
            std::vector<ops::TensorPtr> gradients;
            gradients.reserve(parameters.size());
            for (std::size_t index = 0; index < parameters.size(); ++index) {
                const auto& parameter = parameters[index];
                if (!parameter->grad()) {
                    throw std::runtime_error(
                        "client backward omitted gradient for " +
                        named_parameters[index].name);
                }
                gradients.push_back(parameter->grad());
            }
            optimizer.clip_grad_norm(parameters, options.max_grad_norm);
            optimizer.step(parameters, gradients);
            optimizer.zero_grad(parameters);

            StepMetrics record;
            record.run_id = options.run_id;
            record.client_id = options.client_id;
            record.global_round = round.global_round;
            record.local_step = local_step;
            record.batch_size = batch.batch_size;
            record.sequence_length = batch.sequence_length;
            record.loss = response.loss();
            record.token_accuracy = response.token_accuracy();
            record.split_rpc_bytes = request.ByteSizeLong() + response.ByteSizeLong();
            record.duration_ms = static_cast<std::uint64_t>(elapsed.count());
            metrics.append(record);
            if (observer) {
                observer->on_step(
                    round.global_round, local_step + 1, options.local_steps, response.loss());
            }
        }

        ::sfl::clean::v1::SubmitUpdateRequest submission;
        submission.set_protocol_version(options.protocol_version);
        submission.set_client_id(options.client_id);
        submission.set_global_round(round.global_round);
        submission.set_processed_sequences(processed_sequences);
        const auto updated_state = encode_prefix_lora(named_parameters, tensor_limit);
        for (const auto& tensor : updated_state) *submission.add_prefix_tensors() = tensor;
        rpc.submit(submission);
        completed_any_round = true;
        last_completed_round = round.global_round;
        std::cout << "completed global round " << round.global_round
                  << " after " << processed_sequences << " sequences" << std::endl;
    }

    if (observer) observer->on_phase("COMPLETE", "Training complete");
    std::cout << "coordinator reported DONE" << std::endl;
    return 0;
#endif
}

}  // namespace sflclean
