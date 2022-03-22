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

#include "runtime/cpu.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {

#define REGISTER_OPERATION(__op_type__,                                 \
                           __validate_func_name__,                      \
                           __prepare_func_name__,                       \
                           __execute_func_name__)                       \
  extern bool __validate_func_name__(const core::Operation* operation); \
  extern int __prepare_func_name__(core::Operation* operation);         \
  extern int __execute_func_name__(core::Operation* operation);
namespace operation {
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_CORE_OPERATION_ALL_H__
}  // namespace operation
#undef __NNADAPTER_CORE_OPERATION_ALL_H__
#undef REGISTER_OPERATION

namespace cpu {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  bool num_threads() { return num_threads_; }
  ~Context() {}

 private:
  void* device_{nullptr};
  int num_threads_{0};
};

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  std::string key_value;
  auto key_values = GetKeyValues(properties);
  // CPU_NUM_THREADS
  if (key_values.count(CPU_NUM_THREADS)) {
    num_threads_ = string_parse<int>(key_values[CPU_NUM_THREADS]);
  } else {
    num_threads_ = GetIntFromEnv(CPU_NUM_THREADS, 0);
  }
  NNADAPTER_LOG(INFO) << "num_threads: " << num_threads_;
}

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program() { Clear(); }

  int Validate(const core::Model* model, bool* supported_operations);
  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  void Clear() {
    if (model_.second && model_.first) {
      ClearModel(model_.first);
      delete model_.first;
      model_.first = nullptr;
      model_.second = false;
    }
    operations_.clear();
  }
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  Context* context_{nullptr};
  std::pair<core::Model*, bool> model_;
  std::vector<core::Operation*> operations_;
};

int Program::Validate(const core::Model* model, bool* supported_operations) {
  std::unordered_map<const core::Operation*, size_t> operation_to_index;
  size_t operation_index = 0;
  for (auto& operation : model->operations) {
    operation_to_index[&operation] = operation_index++;
  }
  auto operations = SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Validating " << OperationTypeToString(operation->type)
                      << " ...";
    bool flag = false;
    switch (operation->type) {
#define REGISTER_OPERATION(__op_type__,                  \
                           __validate_func_name__,       \
                           __prepare_func_name__,        \
                           __execute_func_name__)        \
  case NNADAPTER_##__op_type__:                          \
    flag = operation::__validate_func_name__(operation); \
    break;
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_CORE_OPERATION_ALL_H__
#undef REGISTER_OPERATION
      default:
        NNADAPTER_LOG(WARNING) << "Unsupported operation("
                               << OperationTypeToString(operation->type)
                               << ") is found.";
        break;
    }
    supported_operations[operation_to_index[operation]] = flag;
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (!cache->buffer.empty()) {
    // Build from cache
    NNADAPTER_CHECK(!model);
    if (!DeserializeModel(cache->buffer.data(), cache->buffer.size(), &model)) {
      NNADAPTER_LOG(FATAL)
          << "Failed to deserialize the optimized core::Model from a buffer!";
    } else {
      model_.second = true;
      NNADAPTER_VLOG(3)
          << "Deserialize the optimized core::Model from a buffer success.";
    }
    NNADAPTER_VLOG(5) << "Cached model:" << std::endl << Visualize(model);
  } else {
    // Build from model
    model_.second = false;
  }
  model_.first = model;
  if (cache->token && cache->dir) {
    // Serialize core::Model to buffer if cache mode is enabled
    if (cache->buffer.empty()) {
      if (!SerializeModel(model, &cache->buffer)) {
        NNADAPTER_LOG(FATAL)
            << "Failed to serialize the optimized core::Model into a buffer!";
      } else {
        NNADAPTER_VLOG(3)
            << "Serialize the optimized core::Model into a buffer success.";
      }
    }
  }
  operations_ = SortOperationsInTopologicalOrder(model);
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get the new dimensions
    auto& arg = input_arguments[i];
    NNAdapterOperandType new_type;
    arg.access(arg.memory, &new_type);
    // Check whether the rank of input operands have been changed
    const NNAdapterOperandType& old_type =
        model_.first->input_operands[arg.index]->type;
    if (new_type.dimensions.count != old_type.dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int result = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (result != NNADAPTER_NO_ERROR) return result;
  // Set inputs and outputs
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto operand = model_.first->input_operands[arg.index];
    auto type = &operand->type;
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    type->lifetime = NNADAPTER_CONSTANT_REFERENCE;
    operand->buffer = buffer;
    operand->length = GetOperandTypeBufferLength(*type);
  }
  for (auto& operation : operations_) {
    NNADAPTER_VLOG(5) << "Running " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_OPERATION(__op_type__,                            \
                           __validate_func_name__,                 \
                           __prepare_func_name__,                  \
                           __execute_func_name__)                  \
  case NNADAPTER_##__op_type__:                                    \
    NNADAPTER_CHECK(operation::__prepare_func_name__(operation) == \
                    NNADAPTER_NO_ERROR);                           \
    NNADAPTER_CHECK(operation::__execute_func_name__(operation) == \
                    NNADAPTER_NO_ERROR);                           \
    break;
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_CORE_OPERATION_ALL_H__
#undef REGISTER_OPERATION
      default:
        NNADAPTER_LOG(WARNING) << "Unsupported operation("
                               << OperationTypeToString(operation->type)
                               << ") is found.";
        break;
    }
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto operand = model_.first->output_operands[arg.index];
    auto type = &operand->type;
    auto length = GetOperandTypeBufferLength(*type);
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    memcpy(buffer, operand->buffer, length);
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  return NNADAPTER_NO_ERROR;
}

int OpenDevice(void** device) {
  auto d = new Device();
  if (!d) {
    *device = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to open device for cpu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *device = reinterpret_cast<void*>(d);
  return NNADAPTER_NO_ERROR;
}

void CloseDevice(void* device) {
  if (device) {
    auto d = reinterpret_cast<Device*>(device);
    delete d;
  }
}

int CreateContext(void* device, const char* properties, void** context) {
  if (!device || !context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<Device*>(device);
  auto c = new Context(d, properties);
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to create context for cpu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<void*>(c);
  return NNADAPTER_NO_ERROR;
}

void DestroyContext(void* context) {
  if (!context) {
    auto c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int ValidateProgram(void* context,
                    const core::Model* model,
                    bool* supported_operations) {
  NNADAPTER_LOG(INFO) << "Validate program for cpu.";
  if (!context || !model || !supported_operations) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = reinterpret_cast<Context*>(context);
  auto p = new Program(c);
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Validate(model, supported_operations);
  delete p;
  return result;
}

int CreateProgram(void* context,
                  core::Model* model,
                  core::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for cpu.";
  if (!context || !(model || (cache && cache->buffer.size())) || !program) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *program = nullptr;
  auto c = reinterpret_cast<Context*>(context);
  auto p = new Program(c);
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Build(model, cache);
  if (result == NNADAPTER_NO_ERROR) {
    *program = reinterpret_cast<void*>(p);
  }
  return result;
}

void DestroyProgram(void* program) {
  if (program) {
    NNADAPTER_LOG(INFO) << "Destroy program for cpu.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* program,
                   uint32_t input_count,
                   core::Argument* input_arguments,
                   uint32_t output_count,
                   core::Argument* output_arguments) {
  if (!program || !output_arguments || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
  return p->Execute(
      input_count, input_arguments, output_count, output_arguments);
}

}  // namespace cpu
}  // namespace nnadapter

NNADAPTER_EXPORT nnadapter::driver::Device NNADAPTER_AS_SYM2(
    CPU_DEVICE_NAME) = {
    .name = NNADAPTER_AS_STR2(CPU_DEVICE_NAME),
    .vendor = "Paddle",
    .type = NNADAPTER_CPU,
    .version = 1,
    .open_device = nnadapter::cpu::OpenDevice,
    .close_device = nnadapter::cpu::CloseDevice,
    .create_context = nnadapter::cpu::CreateContext,
    .destroy_context = nnadapter::cpu::DestroyContext,
    .validate_program = nnadapter::cpu::ValidateProgram,
    .create_program = nnadapter::cpu::CreateProgram,
    .destroy_program = nnadapter::cpu::DestroyProgram,
    .execute_program = nnadapter::cpu::ExecuteProgram,
};
