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
static const char* NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY = "num_caches";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY =
    "cache_%d_device_name";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY =
    "cache_%d_input_types";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY =
    "cache_%d_output_types";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_INDEXES_KEY =
    "cache_%d_input_indexes";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_INDEXES_KEY =
    "cache_%d_output_indexes";
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

Compilation::Program::~Program() {
  NNADAPTER_CHECK(device_context) << "No device found.";
  device_context->device->DestroyProgram(program);
  if (cache) {
    delete cache;
  }
  if (model) {
    ClearModel(model);
    delete model;
  }
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
  for (auto& operand : operands_) {
    NNADAPTER_CHECK(operand);
    ClearOperand(operand);
    if (operand->buffer) {
      free(operand->buffer);
    }
    delete operand;
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
        bool found = false;
        for (size_t j = 0; j < arguments->size(); j++) {
          if (pos == arguments->at(j).index) {
            arg.memory = arguments->at(j).memory;
            arg.access = arguments->at(j).access;
            found = true;
            break;
          }
        }
        NNADAPTER_CHECK(found) << "No matched argument found!";
      } else {
        for (size_t j = operands->size(); j <= pos; j++) {
          auto operand = new core::Operand();
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
    std::vector<core::Argument> input_args, output_args;
    create_program_arguments(&input_args,
                             programs_[i].input_indexes,
                             input_arguments,
                             &operands_,
                             AccessInputOperand);
    create_program_arguments(&output_args,
                             programs_[i].output_indexes,
                             output_arguments,
                             &operands_,
                             AccessOutputOperand);
    auto result = device_context->device->ExecuteProgram(programs_[i].program,
                                                         input_args.size(),
                                                         input_args.data(),
                                                         output_args.size(),
                                                         output_args.data());
    if (result == NNADAPTER_INVALID_DIMENSIONS) return result;
    NNADAPTER_CHECK_EQ(result, NNADAPTER_NO_ERROR)
        << "Failed to Execute a program for " << i
        << "th compiled program on the device '"
        << device_context->device->GetName() << "'";
  }
  // Serialize the cache models into the file or memory at the first iteration
  if (model_ && CheckCache()) {
    if (!cache_token_.empty() && !cache_dir_.empty()) {
      bool skip = false;
      auto cache_count = programs_.size();
      for (size_t i = 0; i < cache_count; i++) {
        if (!programs_[i].cache->buffer.empty()) continue;
        skip = true;
        auto device_name = programs_[i].device_context->device->GetName();
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
    ClearCache();  // Clear the caches to reduce memory usage
  }
  return NNADAPTER_NO_ERROR;
}

int Compilation::Finish() {
  // Start to build program from model or cache
  completed_ = true;
  programs_.clear();
  if (model_) {
    std::vector<
        std::pair<Context::DeviceContext*,
                  std::tuple<core::Model*, std::vector<int>, std::vector<int>>>>
        models;
    int result = PartitionModel(context_, model_, &models);
    if (result != NNADAPTER_NO_ERROR) {
      return result;
    }
    // Compile these submodels into the programs for the devics and encapsulate
    // the programs into the caches
    auto model_count = models.size();
    programs_.resize(model_count);
    for (size_t i = 0; i < model_count; i++) {
      core::Model* model = nullptr;
      std::vector<int> input_indexes, output_indexes;
      std::tie(model, input_indexes, output_indexes) = models[i].second;
      auto cache = new core::Cache();
      NNADAPTER_CHECK(cache) << "Failed to allocate a cache for the model #"
                             << i << ", out of memory!";
      cache->token = cache_token_.empty() ? nullptr : cache_token_.c_str();
      cache->dir = cache_dir_.empty() ? nullptr : cache_dir_.c_str();
      void* program = nullptr;
      auto device_context = models[i].first;
      NNADAPTER_CHECK_EQ(device_context->device->CreateProgram(
                             device_context->context, model, cache, &program),
                         NNADAPTER_NO_ERROR)
          << "Failed to create a program for " << i
          << "th sub model on the device '" << device_context->device->GetName()
          << "'";
      // Update the types of the submodel inputs and outputs
      auto input_count = model->input_operands.size();
      auto output_count = model->output_operands.size();
      cache->input_types.resize(input_count);
      cache->output_types.resize(output_count);
      for (size_t j = 0; j < input_count; j++) {
        memcpy(&cache->input_types[j],
               &model->input_operands[j]->type,
               sizeof(NNAdapterOperandType));
      }
      for (size_t j = 0; j < output_count; j++) {
        memcpy(&cache->output_types[j],
               &model->output_operands[j]->type,
               sizeof(NNAdapterOperandType));
      }
      // Add into programs
      programs_[i].device_context = device_context;
      programs_[i].model = model;
      programs_[i].cache = cache;
      programs_[i].program = program;
      programs_[i].input_indexes = input_indexes;
      programs_[i].output_indexes = output_indexes;
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
  } else if (CheckCache()) {
    // Compiles the cache models to the programs for the multi-devices
    auto cache_count = programs_.size();
    for (size_t i = 0; i < cache_count; i++) {
      auto device_context = programs_[i].device_context;
      NNADAPTER_CHECK_EQ(
          device_context->device->CreateProgram(device_context->context,
                                                nullptr,
                                                programs_[i].cache,
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

bool Compilation::CheckCache() {
  if (programs_.empty()) return false;
  bool exists = true;
  for (auto& program : programs_) {
    if (!program.cache) {
      exists = false;
      break;
    }
  }
  return exists;
}

void Compilation::ClearCache() {
  for (auto& program : programs_) {
    if (program.cache) {
      delete program.cache;
      program.cache = nullptr;
    }
  }
}

int Compilation::PartitionModel(
    Context* context,
    Model* model,
    std::vector<std::pair<
        Context::DeviceContext*,
        std::tuple<core::Model*, std::vector<int>, std::vector<int>>>>*
        models) {
  // Run the model partition to supports heterogeneous computing on the multiple
  // devices
  models->clear();
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
    std::vector<
        std::pair<int,
                  std::tuple<core::Model*, std::vector<int>, std::vector<int>>>>
        _models_;
    PartitionModelIntoSubmodels(
        &model->model_, supported_operations, &_models_);
    for (auto& _model_ : _models_) {
      auto device_context = context->GetDeviceContext(_model_.first);
      NNADAPTER_CHECK(device_context) << "No device found.";
      models->emplace_back(device_context, _model_.second);
    }
  } else {
    auto device_context = context->GetDeviceContext(0);
    NNADAPTER_CHECK(device_context) << "No device found.";
    std::vector<int> input_indexes, output_indexes;
    for (size_t i = 0; i < model->model_.input_operands.size(); i++) {
      input_indexes.push_back(-i - 1);
    }
    for (size_t i = 0; i < model->model_.output_operands.size(); i++) {
      output_indexes.push_back(-i - 1);
    }
    models->emplace_back(
        device_context,
        std::make_tuple(&model->model_, input_indexes, output_indexes));
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
  uint64_t cache_count = programs_.size();
  NNADAPTER_CHECK(helper->Set(NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY,
                              &cache_count,
                              sizeof(cache_count)));
  for (uint64_t i = 0; i < cache_count; i++) {
    auto& program = programs_[i];
    // cache device name
    auto device_name = program.device_context->device->GetName();
    NNADAPTER_CHECK(helper->Set(
        string_format(NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY, i),
        device_name));
    // input types and indexes
    auto input_count = program.cache->input_types.size();
    if (input_count > 0) {
      // types
      value.resize(input_count * sizeof(NNAdapterOperandType));
      for (size_t j = 0; j < input_count; j++) {
        memcpy(&value[j * sizeof(NNAdapterOperandType)],
               &program.cache->input_types[j],
               sizeof(NNAdapterOperandType));
      }
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY, i),
          value));
      // indexes
      NNADAPTER_CHECK_EQ(input_count, program.input_indexes.size());
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_INDEXES_KEY, i),
          program.input_indexes.data(),
          input_count * sizeof(int)));
    }
    // output types and indexes
    auto output_count = program.cache->output_types.size();
    if (output_count > 0) {
      // types
      value.resize(output_count * sizeof(NNAdapterOperandType));
      for (size_t j = 0; j < output_count; j++) {
        memcpy(&value[j * sizeof(NNAdapterOperandType)],
               &program.cache->output_types[j],
               sizeof(NNAdapterOperandType));
      }
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY, i),
          value));
      // indexes
      NNADAPTER_CHECK_EQ(output_count, program.output_indexes.size());
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_INDEXES_KEY, i),
          program.output_indexes.data(),
          output_count * sizeof(int)));
    }
    // model buffer
    if (!program.cache->buffer.empty()) {
      NNADAPTER_CHECK(helper->Set(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_BUFFER_KEY, i),
          program.cache->buffer));
    }
  }
  auto size = helper->GetSerializedSize();
  buffer->resize(size);
  return helper->Serialize(buffer->data(), size);
}

bool Compilation::Deserialize(void* buffer, uint64_t size) {
  programs_.clear();
  input_types_.clear();
  output_types_.clear();
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
  uint64_t cache_count = 0;
  NNADAPTER_CHECK(helper->Get(
      NNADAPTER_RUNTIME_CACHE_NUM_CACHES_KEY, &cache_count, sizeof(uint64_t)));
  if (cache_count > 0) {
    programs_.resize(cache_count);
    for (size_t i = 0; i < cache_count; i++) {
      auto cache = new core::Cache();
      NNADAPTER_CHECK(cache) << "Failed to allocate a cache for the model #"
                             << i << ", out of memory!";
      cache->token = cache_token_.c_str();
      cache->dir = cache_dir_.c_str();
      cache->input_types.clear();
      cache->output_types.clear();
      cache->buffer.clear();
      // device name
      std::string device_name;
      NNADAPTER_CHECK(helper->Get(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_DEVICE_NAME_KEY, i),
          &device_name));
      programs_[i].device_context =
          context_->GetDeviceContext(device_name.c_str());
      NNADAPTER_CHECK(programs_[i].device_context)
          << "Can't find a device named '" << device_name << "'.";
      // input types and indexes
      if (!helper->Get(
              string_format(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY, i),
              &value)) {
        NNADAPTER_CHECK_EQ(cache_count, 1)
            << "In order to be compatible with the old version of a single "
               "model, we used the wrong key value to get the input types, but "
               "now received "
            << cache_count << " submodels.";
        NNADAPTER_CHECK(helper->Get(
            string_format(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_TYPES_KEY),
            &value));
      }
      auto input_count = value.size() / sizeof(NNAdapterOperandType);
      for (size_t j = 0; j < input_count; j++) {
        cache->input_types.push_back(
            *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + j));
      }
      programs_[i].input_indexes.resize(input_count);
      if (!helper->Get(
              string_format(NNADAPTER_RUNTIME_CACHE_CACHE_INPUT_INDEXES_KEY, i),
              programs_[i].input_indexes.data(),
              input_count * sizeof(int))) {
        NNADAPTER_CHECK_EQ(cache_count, 1)
            << "Only supports missing input indexes for a single model, but "
               "received "
            << cache_count << " submodels.";
        for (size_t j = 0; j < input_count; j++) {
          programs_[i].input_indexes[j] = -j - 1;
        }
      }
      // output types and indexes
      if (!helper->Get(
              string_format(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY, i),
              &value)) {
        NNADAPTER_CHECK_EQ(cache_count, 1)
            << "In order to be compatible with the old version of a single "
               "model, we used the wrong key value to get the output types, "
               "but now received "
            << cache_count << " submodels.";
        helper->Get(
            string_format(NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_TYPES_KEY),
            &value);
      }
      auto output_count = value.size() / sizeof(NNAdapterOperandType);
      for (size_t j = 0; j < output_count; j++) {
        cache->output_types.push_back(
            *(reinterpret_cast<NNAdapterOperandType*>(value.data()) + j));
      }
      programs_[i].output_indexes.resize(output_count);
      if (!helper->Get(string_format(
                           NNADAPTER_RUNTIME_CACHE_CACHE_OUTPUT_INDEXES_KEY, i),
                       programs_[i].output_indexes.data(),
                       output_count * sizeof(int))) {
        NNADAPTER_CHECK_EQ(cache_count, 1)
            << "Only supports missing output indexes for a single model, but "
               "received "
            << cache_count << " submodels.";
        for (size_t j = 0; j < output_count; j++) {
          programs_[i].output_indexes[j] = -j - 1;
        }
      }
      // model buffer
      NNADAPTER_CHECK(helper->Get(
          string_format(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_BUFFER_KEY, i),
          &cache->buffer));
      // Add into programs
      programs_[i].cache = cache;
    }
  }
  return true;
}

}  // namespace runtime
}  // namespace nnadapter
