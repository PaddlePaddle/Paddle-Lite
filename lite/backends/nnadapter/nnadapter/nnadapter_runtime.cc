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

#include "nnadapter_runtime.h"  // NOLINT
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace runtime {

Device::Device(const std::string& name) {
  driver_ = DriverManager::Global().Find(name.c_str());
  if (driver_) {
    driver_->create_context(&context_);
  }
}

Device::~Device() {
  if (driver_ && context_) {
    driver_->destroy_context(context_);
  }
  context_ = nullptr;
  driver_ = nullptr;
}

int Device::CreateProgram(driver::Model* model,
                          driver::Cache* cache,
                          void** program) {
  if (driver_ && context_ && model && program) {
    return driver_->create_program(context_, model, cache, program);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

void Device::DestroyProgram(void* program) {
  if (driver_ && context_ && program) {
    return driver_->destroy_program(context_, program);
  }
}

int Device::ExecuteProgram(void* program,
                           uint32_t input_count,
                           driver::Argument* inputs,
                           uint32_t output_count,
                           driver::Argument* outputs) {
  if (driver_ && context_ && program && outputs && output_count) {
    return driver_->execute_program(
        context_, program, input_count, inputs, output_count, outputs);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

Model::~Model() {
  for (auto& operand : model_.operands) {
    if (operand.type.lifetime == NNADAPTER_CONSTANT && operand.buffer) {
      free(operand.buffer);
    }
    if (operand.type.precision ==
            NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL &&
        operand.type.symm_per_channel_params.scales) {
      free(operand.type.symm_per_channel_params.scales);
    }
  }
}

int Model::AddOperand(const NNAdapterOperandType& type,
                      driver::Operand** operand) {
  model_.operands.emplace_back();
  *operand = &model_.operands.back();
  memcpy(&(*operand)->type, &type, sizeof(NNAdapterOperandType));
  if (type.precision == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL) {
    uint32_t scale_size =
        type.symm_per_channel_params.scale_count * sizeof(float);
    float* scales = reinterpret_cast<float*>(malloc(scale_size));
    NNADAPTER_CHECK(scales)
        << "Failed to allocate the scale buffer for a operand.";
    memcpy(scales, type.symm_per_channel_params.scales, scale_size);
    (*operand)->type.symm_per_channel_params.scales = scales;
  }
  return NNADAPTER_NO_ERROR;
}

int Model::AddOperation(NNAdapterOperationType type,
                        driver::Operation** operation) {
  model_.operations.emplace_back();
  *operation = &model_.operations.back();
  (*operation)->type = type;
  return NNADAPTER_NO_ERROR;
}

int Model::IdentifyInputsAndOutputs(uint32_t input_count,
                                    driver::Operand** inputs,
                                    uint32_t output_count,
                                    driver::Operand** outputs) {
  model_.inputs.resize(input_count);
  for (uint32_t i = 0; i < input_count; i++) {
    model_.inputs[i] = inputs[i];
    model_.inputs[i]->type.lifetime = NNADAPTER_INPUT;
  }
  model_.outputs.resize(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    model_.outputs[i] = outputs[i];
    model_.outputs[i]->type.lifetime = NNADAPTER_OUTPUT;
  }
  return NNADAPTER_NO_ERROR;
}

int Model::Finish() {
  // TODO(hong19860320) model validation
  completed_ = true;
  return NNADAPTER_NO_ERROR;
}

Compilation::Compilation(Model* model,
                         const char* cache_key,
                         void* cache_buffer,
                         size_t cache_length,
                         const char* cache_dir,
                         std::vector<Device*> devices)
    : model_(model), program_(nullptr), devices_(devices), completed_{false} {
  cache_.cache_key = std::string(cache_key);
  cache_.cache_buffer = cache_buffer;
  cache_.cache_length = cache_length;
  cache_.cache_dir = std::string(cache_dir);
}

Compilation::~Compilation() {
  if (program_) {
    auto first_device = GetFirstDevice();
    first_device->DestroyProgram(program_);
  }
}

Device* Compilation::GetFirstDevice() {
  NNADAPTER_CHECK_GT(devices_.size(), 0) << "No device found.";
  auto first_device = devices_[0];
  NNADAPTER_CHECK(first_device->HasDriver())
      << "Driver for device '" << first_device->GetName() << "' not found.";
  return first_device;
}

int Compilation::Execute(std::vector<driver::Argument>* inputs,
                         std::vector<driver::Argument>* outputs) {
  // Execute generated program on target device asynchronously or synchronously
  auto first_device = GetFirstDevice();
  // TODO(hong19860320) support asynchronously execution
  return first_device->ExecuteProgram(program_,
                                      inputs->size(),
                                      &((*inputs)[0]),
                                      outputs->size(),
                                      &((*outputs)[0]));
}

int Compilation::Finish() {
  // Start to build program from model or cache
  completed_ = true;
  auto first_device = GetFirstDevice();
  // TODO(hong19860320) Support the task partition for multi-devices
  return first_device->CreateProgram(&model_->model_, &cache_, &program_);
}

int Execution::SetInput(int32_t index,
                        const uint32_t* dimensions,
                        uint32_t dimension_count,
                        void* buffer,
                        size_t length) {
  driver::Argument* argument = nullptr;
  for (auto& input : inputs_) {
    if (input.index == index) {
      argument = &input;
      break;
    }
  }
  if (!argument) {
    inputs_.emplace_back();
    argument = &inputs_.back();
    argument->index = index;
  }
  argument->dimension_count = dimension_count;
  memcpy(argument->dimensions, dimensions, sizeof(uint32_t) * dimension_count);
  argument->buffer = buffer;
  argument->length = length;
  return NNADAPTER_NO_ERROR;
}

int Execution::SetOutput(int32_t index,
                         const uint32_t* dimensions,
                         uint32_t dimension_count,
                         void* buffer,
                         size_t length) {
  driver::Argument* argument = nullptr;
  for (auto& output : outputs_) {
    if (output.index == index) {
      argument = &output;
      break;
    }
  }
  if (!argument) {
    outputs_.emplace_back();
    argument = &outputs_.back();
    argument->index = index;
  }
  argument->dimension_count = dimension_count;
  memcpy(argument->dimensions, dimensions, sizeof(uint32_t) * dimension_count);
  argument->buffer = buffer;
  argument->length = length;
  return NNADAPTER_NO_ERROR;
}

int Execution::Compute() {
  // TODO(hong19860320) support asynchronously execution
  return compilation_->Execute(&inputs_, &outputs_);
}

DriverManager& DriverManager::Global() {
  static DriverManager driver_manager;
  return driver_manager;
}

DriverManager::DriverManager() {}

DriverManager::~DriverManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < drivers_.size(); i++) {
    void* library = drivers_[i].first;
    if (library) {
      dlclose(library);
    }
  }
  drivers_.clear();
}

size_t DriverManager::Count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return drivers_.size();
}

driver::Driver* DriverManager::At(int index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index >= 0 && index < drivers_.size()) {
    return drivers_[index].second;
  }
  return nullptr;
}

driver::Driver* DriverManager::Find(const char* name) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < drivers_.size(); i++) {
    auto driver = drivers_[i].second;
    if (strcmp(driver->name, name) == 0) {
      return driver;
    }
  }
  // Load if the driver of target device is not registered.
  std::string symbol = std::string(NNADAPTER_AS_STR2(NNADAPTER_DRIVER_PREFIX)) +
                       std::string("_") + name;
  std::string path = std::string("lib") + symbol + std::string(".so");
  void* library = dlopen(path.c_str(), RTLD_NOW);
  if (!library) {
    NNADAPTER_LOG(ERROR) << "Failed to load the nnadapter driver for '" << name
                         << "' from " << path << ", " << dlerror();
    return nullptr;
  }
  auto driver =
      reinterpret_cast<driver::Driver*>(dlsym(library, symbol.c_str()));
  if (!driver) {
    dlclose(library);
    NNADAPTER_LOG(ERROR) << "Failed to find the symbol '" << symbol << "' from "
                         << path << ", " << dlerror();
    return nullptr;
  }
  drivers_.emplace_back(library, driver);
  return driver;
}

}  // namespace runtime
}  // namespace nnadapter
