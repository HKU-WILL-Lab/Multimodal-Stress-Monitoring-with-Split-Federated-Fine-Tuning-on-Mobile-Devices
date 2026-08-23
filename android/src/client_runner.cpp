#include "sfl/client_runner.h"

#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/tokenizer_gemma.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/model_registry.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"
#include "sfl/client_options.h"
#include "sfl/lora_state.h"
#include "sfl/metrics.h"
#include "sfl/tensor_codec.h"
#include "sfl/wiki_text_stream.h"
#include "sfl_clean.grpc.pb.h"

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

void validate_gemma_3_270m(const std::string& model_dir,
                           const ops::GemmaTextConfig& config) {
    const auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
    if (spec.family != ops::ModelFamily::Gemma ||
        config.vocab_size != 262144 || config.hidden_size != 640 ||
        config.intermediate_size != 2048 || config.num_hidden_layers != 18 ||
        config.num_attention_heads != 4 || config.num_key_value_heads != 1 ||
        config.head_dim != 256) {
        throw std::invalid_argument(
            "this client accepts only the text Gemma 3 270M architecture");
    }
}

std::unordered_map<std::string, std::string> prefix_weight_mapping(int layer_count) {
    auto full = ops::GemmaKeyMapper::generate_gemma_mapping(layer_count);
    std::unordered_map<std::string, std::string> prefix;
    for (auto& item : full) {
        if (item.first == "embed_tokens.weight") {
            prefix.emplace(std::move(item));
            continue;
        }
        if (item.first.rfind("layers.", 0) != 0) continue;
        const auto start = std::string("layers.").size();
        const auto dot = item.first.find('.', start);
        if (dot == std::string::npos) throw std::logic_error("invalid Gemma key mapping");
        const int layer = std::stoi(item.first.substr(start, dot - start));
        if (layer < layer_count) prefix.emplace(std::move(item));
    }
    return prefix;
}

ops::TensorPtr int32_tensor(const std::vector<std::int64_t>& values,
                            std::uint32_t batch,
                            std::uint32_t sequence,
                            const char* field) {
    std::vector<std::int32_t> converted;
    converted.reserve(values.size());
    for (const auto value : values) {
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(std::string(field) + " value exceeds MobileFineTuner int32 range");
        }
        converted.push_back(static_cast<std::int32_t>(value));
    }
    return std::make_shared<ops::Tensor>(
        std::vector<std::int64_t>{batch, sequence},
        converted.data(), ops::DType::kInt32, ops::kCPU);
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
    auto config = ops::GemmaTextConfig::from_pretrained(options.model_dir);
    validate_gemma_3_270m(options.model_dir, config);
    if (options.cut_layer >= static_cast<std::uint32_t>(config.num_hidden_layers)) {
        throw std::invalid_argument("cut-layer must be positive and below the final decoder layer");
    }

    ops::GemmaModel model(
        config, ops::GemmaHiddenSpanConfig::prefix(static_cast<int>(options.cut_layer)));
    ops::SafeTensorsModelReader reader(options.model_dir);
    reader.parse_headers();
    ops::SafeTensorsLoadOptions load_options;
    load_options.verbose = false;
    const auto mapping = prefix_weight_mapping(static_cast<int>(options.cut_layer));
    auto loaded = reader.load_tensors_mapped(mapping, load_options);
    if (loaded.size() != mapping.size()) {
        throw std::runtime_error("model checkpoint did not contain every required prefix weight");
    }
    for (auto& item : loaded) model.assign_weight(item.first, item.second);

    ops::GemmaLoraSpec lora;
    lora.rank = static_cast<int>(options.lora_rank);
    lora.alpha = options.lora_alpha;
    lora.dropout = 0.0F;
    for (std::uint32_t layer = 0; layer < options.cut_layer; ++layer) {
        lora.layers.push_back(static_cast<int>(layer));
    }
    ops::GemmaLoraInjector injector;
    injector.inject(model, lora);
    auto named_parameters = prefix_lora_parameters(model, static_cast<int>(options.cut_layer));
    std::vector<ops::TensorPtr> parameters;
    parameters.reserve(named_parameters.size());
    for (const auto& item : named_parameters) parameters.push_back(item.tensor);

    ops::AdamConfig adam_config(options.learning_rate, 0.9F, 0.999F, 1.0e-8F,
                                0.0F, options.max_grad_norm, false);
    ops::Adam optimizer(adam_config);

    auto tokenizer_config = ops::GemmaTokenizerConfig::from_pretrained(options.model_dir);
    ops::GemmaTokenizer tokenizer(tokenizer_config);
    tokenizer.load();
    WikiTextStream stream(
        options.dataset_path, options.client_index, options.client_count,
        tokenizer.get_eos_token_id(), tokenizer.get_pad_token_id(),
        [&tokenizer](const std::string& line) {
            return tokenizer.encode(line, false, 0, false);
        });

    const std::size_t tensor_limit =
        static_cast<std::size_t>(options.max_message_bytes);
    auto initial_state = encode_prefix_lora(named_parameters, tensor_limit);
    RpcClients rpc(options);
    JsonlMetricWriter metrics(options.metrics_path);

    bool completed_any_round = false;
    std::uint64_t last_completed_round = 0;
    std::uint64_t last_server_step = 0;
    while (true) {
        RoundView round = rpc.fetch_or_bootstrap(last_completed_round, initial_state);
        if (round.state == ::sfl::clean::v1::DONE) break;
        if (round.state == ::sfl::clean::v1::WAIT) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.poll_interval_ms));
            continue;
        }
        if (completed_any_round && round.global_round <= last_completed_round) {
            throw std::runtime_error("coordinator returned TRAIN for an already completed round");
        }
        load_prefix_lora(round.prefix_tensors, named_parameters, tensor_limit);

        std::uint64_t processed_sequences = 0;
        for (std::uint32_t local_step = 0; local_step < options.local_steps; ++local_step) {
            TokenBatch batch = stream.next_batch(options.batch_size, options.sequence_length);
            if (batch.batch_size == 0) {
                throw std::runtime_error("assigned dataset partition was exhausted before local training finished");
            }
            processed_sequences += batch.processed_sequences;
            auto token_ids = int32_tensor(batch.token_ids, batch.batch_size,
                                          batch.sequence_length, "token_ids");
            auto attention_mask = int32_tensor(batch.attention_mask, batch.batch_size,
                                               batch.sequence_length, "attention_mask");
            auto activation = model.forward_hidden_states(token_ids, attention_mask);

            ::sfl::clean::v1::SplitStepRequest request;
            request.set_protocol_version(options.protocol_version);
            request.set_client_id(options.client_id);
            request.set_global_round(round.global_round);
            request.set_local_step(local_step);
            request.set_cut_layer(options.cut_layer);
            *request.mutable_activation() = encode_float32_tensor(
                "boundary_activation", activation, tensor_limit);
            *request.mutable_token_ids() = encode_int64_tensor(
                "token_ids", batch.token_ids,
                {batch.batch_size, batch.sequence_length}, tensor_limit);
            *request.mutable_attention_mask() = encode_int64_tensor(
                "attention_mask", batch.attention_mask,
                {batch.batch_size, batch.sequence_length}, tensor_limit);

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
            for (const auto& parameter : parameters) {
                if (!parameter->grad()) {
                    throw std::runtime_error("prefix backward omitted a LoRA gradient");
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

    std::cout << "coordinator reported DONE" << std::endl;
    return 0;
}

}  // namespace sflclean
