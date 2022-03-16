// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/compilation.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "optimizer/partition_model_into_submodels.h"
#include "utility/cache.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace runtime {

static const char* NNADAPTER_RUNTIME_CACHE_FILE_EXTENSION = ".nnc";
static const char* NNADAPTER_RUNTIME_CACHE_INPUT_TYPES_KEY = "input_types";
static const char* NNADAPTER_RUNTIME_CACHE_OUTPUT_TYPES_KEY = "output_types";
static const char* NNADAPTER_RUNTIME_CACHE_INPUT_INDEXES_KEY = "input_indexes";
static const char* NNADAPTER_RUNTIME_CACHE_OUTPUT_INDEXES_KEY =
    "output_indexes";
static const char* NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY = "num_caches";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY =
    "cache_%d_device_name";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY =
    "cache_%d_input_types";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY =
    "cache_%d_output_types";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_BUFFER_KEY =
    "cache_%d_model_buffer";

void* AccessInputOperand(void* memory, NNAdapterOperandType* type) {
  NNADAPTER_CHECK(memory);
  NNADAPTER_CHECK(type);
  auto operand = static_cast<core::Operand*>(memory);
  CopyOperandType(type, operand->type);
  return operand->buffer;
}

void* AccessOutputOperand(void* memory, NNAdapterOperandType* type) {
  NNADAPTER_CHECK(memory);
  NNADAPTER_CHECK(type);
  auto operand = static_cast<core::Operand*>(memory);
  CopyOperandType(&operand->type, *type);
  auto length = GetOperandTypeBufferLength(operand->type);
  if (operand->length < length) {
    if (operand->buffer) {
      free(operand->buffer);
    }
    operand->buffer = malloc(length);
    NNADAPTER_CHECK(operand->buffer) << "Failed to allocate " << length
                                     << " bytes, out of memory!";
    operand->length = length;
  }
  return operand->buffer;
}

Compilation::Compilation(Model* model,
                         const char* cache_token,
                         void* cache_buffer,
                         uint32_t cache_length,
                         const char* cache_dir,
                         Context* context)
    : model_(model),
      cache_token_(cache_token),
      cache_dir_(cache_dir),
      context_(context),
      completed_(false) {
  // Deserialize the cache models from the file or memory
  if (!model_) {
    std::vector<uint8_t> buffer;
    if (!cache_token_.empty() && !cache_dir_.empty() &&
        (!cache_buffer || !cache_length)) {
      std::string path = cache_dir_ + "/" + cache_token_ +
                         std::string(NNADAPTER_RUNTIME_CACHE_FILE_EXTENSION);
      if (ReadFile(path, &buffer)) {
        NNADAPTER_LOG(INFO) << "Read the cache file " << path << " success.";
        cache_buffer = buffer.data();
        cache_length = buffer.size();
      }
    }
    if (cache_buffer && cache_length) {
      if (Deserialize(cache_buffer, cache_length)) {
        NNADAPTER_LOG(INFO)
            << "Deserialize the cache models from memory success.";
      }
    }
  }
}

Compilation::~Compilation() {
  for (size_t i = 0; i < programs_.size(); i++) {
    auto device_context = programs_[i].device_context;
    NNADAPTER_CHECK(device_context) << "No device found.";
    device_context->device->DestroyProgram(programs_[i].program);
  }
  for (size_t i = 0; i < operands_.size(); i++) {
    auto operand = operands_[i];
    NNADAPTER_CHECK(operand);
    if (operand->buffer) {
      free(operand->buffer);
    }
    free(operand);
  }
  for (auto& model : models_) {
    NNADAPTER_CHECK(model.second);
    nnadapter::ClearModel(model.second);
    delete model.second;
  }
}

int Compilation::Execute(std::vector<core::Argument>* input_arguments,
                         std::vector<core::Argument>* output_arguments) {
  auto create_program_arguments = [&](
      std::vector<core::Argument>* args,
      const std::vector<int>& indexes,
      std::vector<core::Argument>* arguments,
      std::vector<core::Operand*>* operands,
      void* (*access)(void* memory, NNAdapterOperandType* type)) {
    for (size_t i = 0; i < indexes.size(); i++) {
      core::Argument arg;
      arg.index = i;
      auto pos = indexes[i];
      if (pos < 0) {
        pos = -pos - 1;
        arg.memory = arguments->at(pos).memory;
        arg.access = arguments->at(pos).access;
      } else {
        for (size_t j = operands->size(); j <= pos; j++) {
          auto operand =
              static_cast<core::Operand*>(malloc(sizeof(core::Operand)));
          NNADAPTER_CHECK(operand)
              << "Failed to allocate memory for a operand, out of memory!";
          memset(operand, 0, sizeof(core::Operand));
          operands->push_back(operand);
        }
        arg.memory = operands->at(pos);
        arg.access = access;
      }
      args->push_back(arg);
    }
  };
  // Executes the compiled programs on the multi-devices asynchronously or
  // synchronously
  // TODO(hong19860320) Supports asynchronously execution in future.
  for (size_t i = 0; i < programs_.size(); i++) {
    auto device_context = programs_[i].device_context;
    std::vector<core::Argument> program_input_arguments,
        program_output_arguments;
    create_program_arguments(&program_input_arguments,
                             input_indexes_[i],
                             input_arguments,
                             &operands_,
                             AccessInputOperand);
    create_program_arguments(&program_output_arguments,
                             output_indexes_[i],
                             output_arguments,
                             &operands_,
                             AccessOutputOperand);
    auto result =
        device_context->device->ExecuteProgram(programs_[i].program,
                                               program_input_arguments.size(),
                                               program_input_arguments.data(),
                                               program_output_arguments.size(),
                                               program_output_arguments.data());
    if (result == NNADAPTER_INVALID_DIMENSIONS) return result;
    NNADAPTER_CHECK_EQ(result, NNADAPTER_NO_ERROR)
        << "Failed to Execute a program for " << i
        << "th compiled program on the device '"
        << device_context->device->GetName() << "'";
  }
  // Serialize the cache models into the file or memory at the first iteration
  if (model_ && !caches_.empty()) {
    if (!cache_token_.empty() && !cache_dir_.empty()) {
      bool skip = false;
      for (size_t i = 0; i < caches_.size(); i++) {
        if (!caches_[i].cache.buffer.empty()) continue;
        skip = true;
        auto device_name = caches_[i].device_context->device->GetName();
        NNADAPTER_LOG(WARNING) << "The " << i << "th device '" << device_name
                               << "' doesn't support model cache!";
        break;
      }
      if (!skip) {
        std::vector<uint8_t> buffer;
        if (Serialize(&buffer)) {
          NNADAPTER_LOG(INFO)
              << "Serialize the cache models into memory success.";
          std::string path =
              cache_dir_ + "/" + cache_token_ +
              std::string(NNADAPTER_RUNTIME_CACHE_FILE_EXTENSION);
          if (WriteFile(path, buffer)) {
            NNADAPTER_LOG(INFO) << "Write the cache file " << path
                                << " success.";
          }
        }
      }
    }
    caches_.clear();  // Clear the caches to reduce memory usage
  }
  return NNADAPTER_NO_ERROR;
}

int Compilation::Finish() {
  // Start to build program from model or cache
  completed_ = true;
  programs_.clear();
  if (model_) {
    input_types_.clear();
    output_types_.clear();
    caches_.clear();
    int result = PartitionModel(
        context_, model_, &models_, &input_indexes_, &output_indexes_);
    if (result != NNADAPTER_NO_ERROR) {
      return result;
    }
    // Compile these submodels into the programs for the devics and encapsulate
    // the programs into the caches
    auto model_count = models_.size();
    caches_.resize(model_count);
    programs_.resize(model_count);
    for (size_t i = 0; i < model_count; i++) {
      auto device_context = models_[i].first;
      auto model = models_[i].second;
      caches_[i].device_context = device_context;
      caches_[i].cache.token =
          cache_token_.empty() ? nullptr : cache_token_.c_str();
      caches_[i].cache.dir = cache_dir_.empty() ? nullptr : cache_dir_.c_str();
      NNADAPTER_CHECK_EQ(
          device_context->device->CreateProgram(device_context->context,
                                                model,
                                                &caches_[i].cache,
                                                &programs_[i].program),
          NNADAPTER_NO_ERROR)
          << "Failed to create a program for " << i
          << "th sub model on the device '" << device_context->device->GetName()
          << "'";
      programs_[i].device_context = device_context;
      // Update the types of the submodel inputs and outputs
      auto input_count = model->input_operands.size();
      auto output_count = model->output_operands.size();
      caches_[i].cache.input_types.resize(input_count);
      caches_[i].cache.output_types.resize(output_count);
      for (size_t j = 0; j < input_count; j++) {
        memcpy(&caches_[i].cache.input_types[j],
               &model->input_operands[j]->type,
               sizeof(NNAdapterOperandType));
      }
      for (size_t j = 0; j < output_count; j++) {
        memcpy(&caches_[i].cache.output_types[j],
               &model->output_operands[j]->type,
               sizeof(NNAdapterOperandType));
      }
    }
    // Update the types of the model inputs and outputs
    auto input_count = model_->model_.input_operands.size();
    auto output_count = model_->model_.output_operands.size();
    input_types_.resize(input_count);
    output_types_.resize(output_count);
    for (uint32_t i = 0; i < input_count; i++) {
      memcpy(&input_types_[i],
             &model_->model_.input_operands[i]->type,
             sizeof(NNAdapterOperandType));
    }
    for (uint32_t i = 0; i < output_count; i++) {
      memcpy(&output_types_[i],
             &model_->model_.output_operands[i]->type,
             sizeof(NNAdapterOperandType));
    }
  } else if (!caches_.empty()) {
    // Compiles the cache models to the programs for the multi-devices
    auto num_caches = caches_.size();
    programs_.resize(num_caches);
    for (size_t i = 0; i < num_caches; i++) {
      auto device_context = caches_[i].device_context;
      programs_[i].device_context = device_context;
      NNADAPTER_CHECK_EQ(
          device_context->device->CreateProgram(device_context->context,
                                                nullptr,
                                                &caches_[i].cache,
                                                &programs_[i].program),
          NNADAPTER_NO_ERROR)
          << "Failed to create a program for " << i
          << "th cache model on the device '"
          << device_context->device->GetName() << "'";
    }
  } else {
    NNADAPTER_LOG(WARNING)
        << "Failed to create a program, No model and cache is provided.";
    return NNADAPTER_INVALID_PARAMETER;
  }
  return NNADAPTER_NO_ERROR;
}

int Compilation::QueryInputsAndOutputs(uint32_t* input_count,
                                       NNAdapterOperandType** input_types,
                                       uint32_t* output_count,
                                       NNAdapterOperandType** output_types) {
  if (!input_count || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *input_count = static_cast<uint32_t>(input_types_.size());
  if (input_types) {
    for (uint32_t i = 0; i < *input_count; i++) {
      input_types[i] = &input_types_[i];
    }
  }
  *output_count = static_cast<uint32_t>(output_types_.size());
  if (output_types) {
    for (uint32_t i = 0; i < *output_count; i++) {
      output_types[i] = &output_types_[i];
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Compilation::PartitionModel(
    Context* context,
    Model* model,
    std::vector<std::pair<Context::DeviceContext*, core::Model*>>* models,
    std::vector<std::vector<int>>* input_indexes,
    std::vector<std::vector<int>>* output_indexes) {
  // Run the model partition to supports heterogeneous computing on the multiple
  // devices
  models->clear();
  input_indexes->clear();
  output_indexes->clear();
  auto device_count = context->GetDeviceCount();
  NNADAPTER_CHECK_GE(device_count, 1) << "No device found.";
  if (device_count > 1) {
    auto operation_count = model->model_.operations.size();
    std::unique_ptr<bool[]> flags(new bool[operation_count]);
    std::fill(flags.get(), flags.get() + operation_count, false);
    std::vector<std::pair<int, std::unordered_set<core::Operation*>>>
        supported_operations(device_count);
    for (size_t i = 0; i < device_count; i++) {
      auto device_context = context->GetDeviceContext(i);
      NNADAPTER_CHECK(device_context);
      std::unique_ptr<bool[]> _flags_(new bool[operation_count]);
      auto result =
          model->GetSupportedOperations(device_context, _flags_.get());
      if (result != NNADAPTER_NO_ERROR) {
        return result;
      }
      supported_operations[i].first = i;
      size_t operation_index = 0;
      for (auto& operation : model->model_.operations) {
        if (_flags_[operation_index] && !flags[operation_index]) {
          flags[operation_index] = true;
          // Only the operations which are not supported by the previous devices
          // are added.
          supported_operations[i].second.insert(&operation);
        }
        operation_index++;
      }
    }
    // Check for operators not supported in these devices
    size_t operation_index = 0;
    for (auto& operation : model->model_.operations) {
      if (flags[operation_index++]) continue;
      NNADAPTER_LOG(FATAL) << "None of these " << device_count
                           << " devices support "
                           << OperationTypeToString(operation.type) << "!";
    }
    std::vector<std::pair<int, core::Model*>> _models_;
    PartitionModelIntoSubmodels(&model->model_,
                                supported_operations,
                                &_models_,
                                input_indexes,
                                output_indexes);
    for (auto& _model_ : _models_) {
      auto device_context = context->GetDeviceContext(_model_.first);
      NNADAPTER_CHECK(device_context) << "No device found.";
      models->emplace_back(device_context, _model_.second);
    }
  } else {
    auto device_context = context->GetDeviceContext(0);
    NNADAPTER_CHECK(device_context) << "No device found.";
    models->push_back(std::pair<Context::DeviceContext*, core::Model*>(
        device_context, &model->model_));
    input_indexes->resize(1);
    output_indexes->resize(1);
    for (size_t i = 0; i < model->model_.input_operands.size(); i++) {
      input_indexes->at(0).push_back(-i - 1);
    }
    for (size_t i = 0; i < model->model_.output_operands.size(); i++) {
      output_indexes->at(0).push_back(-i - 1);
    }
  }
  return NNADAPTER_NO_ERROR;
}

bool Compilation::Serialize(std::vector<uint8_t>* buffer) {
  auto helper = std::make_shared<nnadapter::Cache>();
  std::vector<uint8_t> value;
  // Serialize the model input types
  auto input_count = input_types_.size();
  if (input_count > 0) {
    value.resize(input_count * sizeof(NNAdapterOperandType));
    for (size_t i = 0; i < input_count; i++) {
      memcpy(&value[i * sizeof(NNAdapterOperandType)],
             &input_types_[i],
             sizeof(NNAdapterOperandType));
    }
    NNADAPTER_CHECK(
        helper->Set(NNADAPTER_RUNTIME_CACHE_INPUT_TYPES_KEY, value));
  }
  // Serialize the model output types
  auto output_count = output_types_.size();
  if (output_count > 0) {
    value.resize(output_count * sizeof(NNAdapterOperandType));
    for (size_t i = 0; i < output_count; i++) {
      memcpy(&value[i * sizeof(NNAdapterOperandType)],
             &output_types_[i],
             sizeof(NNAdapterOperandType));
    }
    NNADAPTER_CHECK(
        helper->Set(NNADAPTER_RUNTIME_CACHE_OUTPUT_TYPES_KEY, value));
  }
  // Serialize all of device-specific compiled binary program
  uint64_t num_caches = caches_.size();
  NNADAPTER_CHECK(helper->Set(
      NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY, &num_caches, sizeof(num_caches)));
  for (uint64_t i = 0; i < num_caches; i++) {
    // cache device name
    auto device_name = caches_[i].device_context->device->GetName();
    NNADAPTER_CHECK(helper->Set(
        string_format(NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY, i),
        device_name,
        strlen(device_name)));
    // cache input types
    auto input_count = caches_[i].cache.input_types.size();
    if (input_count > 0) {
      value.resize(input_count * sizeof(NNAdapterOperandType));
      for (size_t j = 0; j < input_count; j++) {
        memcpy(&value[j * sizeof(NNAdapterOperandType)],
               &caches_[i].cache.input_types[j],
               sizeof(NNAdapterOperandType));
      }
      NNADAPTER_CHECK(
          helper->Set(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY, value));
    }
    // cache output types
    auto output_count = caches_[i].cache.output_types.size();
    if (output_count > 0) {
      value.resize(output_count * sizeof(NNAdapterOperandType));
      for (size_t j = 0; j < output_count; j++) {
        memcpy(&value[j * sizeof(NNAdapterOperandType)],
               &caches_[i].cache.output_types[j],
               sizeof(NNAdapterOperandType));
      }
      NNADAPTER_CHECK(
          helper->Set(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY, value));
    }
    // cache model buffer
    if (!caches_[i].cache.buffer.empty()) {
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_BUFFER_KEY, i),
          caches_[i].cache.buffer));
    }
  }
  auto size = helper->GetSerializedSize();
  buffer->resize(size);
  return helper->Serialize(buffer->data(), size);
}

bool Compilation::Deserialize(void* buffer, uint64_t size) {
  input_types_.clear();
  output_types_.clear();
  caches_.clear();
  auto helper = std::make_shared<nnadapter::Cache>();
  if (!helper->Deserialize(buffer, size)) {
    return false;
  }
  std::vector<uint8_t> value;
  // Parsing the model input types
  NNADAPTER_CHECK(helper->Get(NNADAPTER_RUNTIME_CACHE_INPUT_TYPES_KEY, &value));
  auto input_count = value.size() / sizeof(NNAdapterOperandType);
  for (size_t i = 0; i < input_count; i++) {
    input_types_.push_back(
        *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + i));
  }
  // Parsing the model output types
  NNADAPTER_CHECK(
      helper->Get(NNADAPTER_RUNTIME_CACHE_OUTPUT_TYPES_KEY, &value));
  auto output_count = value.size() / sizeof(NNAdapterOperandType);
  for (size_t i = 0; i < output_count; i++) {
    output_types_.push_back(
        *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + i));
  }
  // Parsing all of device-specific compiled binary program
  NNADAPTER_CHECK(helper->Get(NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY, &value));
  NNADAPTER_CHECK_EQ(value.size(), sizeof(uint64_t));
  auto num_caches = *(reinterpret_cast<uint64_t*>(value.data()));
  if (num_caches > 0) {
    caches_.resize(num_caches);
    for (size_t i = 0; i < num_caches; i++) {
      caches_[i].cache.token = cache_token_.c_str();
      caches_[i].cache.dir = cache_dir_.c_str();
      caches_[i].cache.input_types.clear();
      caches_[i].cache.output_types.clear();
      caches_[i].cache.buffer.clear();
      // cache device name
      NNADAPTER_CHECK(helper->Get(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY, i),
          &value));
      std::string device_name(
          reinterpret_cast<char*>(value.data()),
          reinterpret_cast<char*>(value.data()) + value.size());
      caches_[i].device_context =
          context_->GetDeviceContext(device_name.c_str());
      NNADAPTER_CHECK(caches_[i].device_context)
          << "Can't find a device named '" << device_name << "'.";
      // cache input types
      NNADAPTER_CHECK(
          helper->Get(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY, &value));
      auto input_count = value.size() / sizeof(NNAdapterOperandType);
      for (size_t j = 0; j < input_count; j++) {
        caches_[i].cache.input_types.push_back(
            *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + j));
      }
      // cache output types
      NNADAPTER_CHECK(
          helper->Get(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY, &value));
      auto output_count = value.size() / sizeof(NNAdapterOperandType);
      for (size_t j = 0; j < output_count; j++) {
        caches_[i].cache.output_types.push_back(
            *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + j));
      }
      // cache model buffer
      NNADAPTER_CHECK(helper->Get(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_BUFFER_KEY, i),
          &caches_[i].cache.buffer));
    }
  }
  return true;
}

}  // namespace runtime
}  // namespace nnadapter
