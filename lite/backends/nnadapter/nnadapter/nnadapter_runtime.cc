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
  driver_ = Driver::Global().Find(name.c_str());
  if (driver_) {
    driver_->createContext(&context_);
  }
}

Device::~Device() {
  if (driver_ && context_) {
    driver_->destroyContext(context_);
  }
  context_ = nullptr;
  driver_ = nullptr;
}

int Device::createModelFromGraph(driver::Graph* graph, void** model) {
  if (driver_ && context_ && graph && model) {
    return driver_->createModelFromGraph(context_, graph, model);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

int Device::createModelFromCache(void* buffer, size_t length, void** model) {
  if (driver_ && context_ && buffer && length) {
    return driver_->createModelFromCache(context_, buffer, length, model);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

void Device::destroyModel(void* model) {
  if (driver_ && context_ && model) {
    return driver_->destroyModel(context_, model);
  }
}

int Device::runModelSync(void* model,
                         uint32_t inputCount,
                         driver::Operand** inputs,
                         uint32_t outputCount,
                         driver::Operand** outputs) {
  if (driver_ && context_ && model && outputs && outputCount) {
    return driver_->runModelSync(
        context_, model, inputCount, inputs, outputCount, outputs);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

int Device::runModelAsync(void* model,
                          uint32_t inputCount,
                          driver::Operand** inputs,
                          uint32_t outputCount,
                          driver::Operand** outputs) {
  if (driver_ && context_ && outputs && outputCount) {
    return driver_->runModelAsync(
        context_, model, inputCount, inputs, outputCount, outputs);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

Graph::~Graph() {
  for (auto& operand : graph_.operands) {
    if (operand.type.lifetime == NNADAPTER_CONSTANT && operand.buffer) {
      free(operand.buffer);
    }
    if (operand.type.precision ==
            NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL &&
        operand.type.symmPerChannelParams.scales) {
      free(operand.type.symmPerChannelParams.scales);
    }
  }
}

int Graph::addOperand(const NNAdapterOperandType& type,
                      driver::Operand** operand) {
  graph_.operands.emplace_back();
  *operand = &graph_.operands.back();
  memcpy(&(*operand)->type, &type, sizeof(NNAdapterOperandType));
  if (type.precision == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL) {
    uint32_t scaleSize = type.symmPerChannelParams.scaleCount * sizeof(float);
    float* scales = reinterpret_cast<float*>(malloc(scaleSize));
    NNADAPTER_CHECK(scales)
        << "Failed to allocate the scale buffer for a operand.";
    memcpy(scales, type.symmPerChannelParams.scales, scaleSize);
    (*operand)->type.symmPerChannelParams.scales = scales;
  }
  return NNADAPTER_NO_ERROR;
}

int Graph::addOperation(NNAdapterOperationType type,
                        driver::Operation** operation) {
  graph_.operations.emplace_back();
  *operation = &graph_.operations.back();
  (*operation)->type = type;
  return NNADAPTER_NO_ERROR;
}

int Graph::identifyInputsAndOutputs(uint32_t inputCount,
                                    driver::Operand** inputs,
                                    uint32_t outputCount,
                                    driver::Operand** outputs) {
  graph_.inputs.resize(inputCount);
  for (uint32_t i = 0; i < inputCount; i++) {
    graph_.inputs[i] = inputs[i];
    graph_.inputs[i]->type.lifetime = NNADAPTER_INPUT;
  }
  graph_.outputs.resize(outputCount);
  for (uint32_t i = 0; i < outputCount; i++) {
    graph_.outputs[i] = outputs[i];
    graph_.outputs[i]->type.lifetime = NNADAPTER_OUTPUT;
  }
  return NNADAPTER_NO_ERROR;
}

int Graph::finish() {
  // TODO(hong19860320) model validation
  completed_ = true;
  return NNADAPTER_NO_ERROR;
}

Model::Model(void* buffer,
             size_t length,
             uint32_t inputCount,
             const NNAdapterOperandType** inputTypes,
             uint32_t outputCount,
             const NNAdapterOperandType** outputTypes,
             std::vector<Device*> devices)
    : graph_(nullptr),
      model_(nullptr),
      cached_(true),
      buffer_(buffer),
      length_(length),
      devices_(devices),
      completed_{false} {
  // Create a fake model, and add input&output operands
  graph_ = new Graph();
  std::vector<driver::Operand*> is(inputCount);
  for (uint32_t i = 0; i < inputCount; i++) {
    graph_->addOperand(*inputTypes[i], &is[i]);
  }
  std::vector<driver::Operand*> os(outputCount);
  for (uint32_t i = 0; i < outputCount; i++) {
    graph_->addOperand(*outputTypes[i], &os[i]);
  }
  graph_->identifyInputsAndOutputs(is.size(), &is[0], os.size(), &os[0]);
  graph_->finish();
}

Model::~Model() {
  if (cached_ && graph_) {
    delete graph_;
  }
  if (model_) {
    auto first_device = firstDevice();
    first_device->destroyModel(model_);
  }
}

Device* Model::firstDevice() {
  NNADAPTER_CHECK_GT(devices_.size(), 0) << "No device found.";
  auto first_device = devices_[0];
  NNADAPTER_CHECK(first_device->hasDriver())
      << "Driver for device '" << first_device->getName() << "' not found.";
  return first_device;
}

int Model::setCaching(const char* cacheDir, const uint8_t* token) {
  NNADAPTER_NO_ERROR;
}

int Model::setInput(int32_t index,
                    const uint32_t* dimensions,
                    uint32_t dimensionCount,
                    void* buffer,
                    size_t length) {
  auto& inputs = graph_->graph_.inputs;
  if (index < 0 || index >= inputs.size()) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  inputs[index]->type.dimensionCount = dimensionCount;
  memcpy(inputs[index]->type.dimensions,
         dimensions,
         dimensionCount * sizeof(uint32_t));
  inputs[index]->buffer = buffer;
  inputs[index]->length = length;
  return NNADAPTER_NO_ERROR;
}

int Model::setOutput(int32_t index,
                     const uint32_t* dimensions,
                     uint32_t dimensionCount,
                     void* buffer,
                     size_t length) {
  auto& outputs = graph_->graph_.outputs;
  if (index < 0 || index >= outputs.size()) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  outputs[index]->type.dimensionCount = dimensionCount;
  memcpy(outputs[index]->type.dimensions,
         dimensions,
         dimensionCount * sizeof(uint32_t));
  outputs[index]->buffer = buffer;
  outputs[index]->length = length;
  return NNADAPTER_NO_ERROR;
}

int Model::run(Event** event) {
  // TODO(hong19860320) add callback
  *event = new Event();
  if (!*event) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  // Invoke the driver of target device to compute model asynchronously or
  // synchronously
  auto first_device = firstDevice();
  std::vector<driver::Operand*>& inputs = graph_->graph_.inputs;
  std::vector<driver::Operand*>& outputs = graph_->graph_.outputs;
  if (event) {
    return first_device->runModelAsync(
        model_, inputs.size(), &inputs[0], outputs.size(), &outputs[0]);
  }
  return first_device->runModelSync(
      model_, inputs.size(), &inputs[0], outputs.size(), &outputs[0]);
}

int Model::finish() {
  // Invoke the driver of target device to build from cache or model
  completed_ = true;
  auto first_device = firstDevice();
  if (cached_) {
    return first_device->createModelFromCache(buffer_, length_, &model_);
  }
  // TODO(hong19860320) Support the task partition for multi-devices
  return first_device->createModelFromGraph(&graph_->graph_, &model_);
}

int Execution::setInput(int32_t index,
                        const uint32_t* dimensions,
                        uint32_t dimensionCount,
                        void* buffer,
                        size_t length) {
  return model_->setInput(index, dimensions, dimensionCount, buffer, length);
}

int Execution::setOutput(int32_t index,
                         const uint32_t* dimensions,
                         uint32_t dimensionCount,
                         void* buffer,
                         size_t length) {
  return model_->setOutput(index, dimensions, dimensionCount, buffer, length);
}

int Execution::run(Event** event) { return model_->run(event); }

Driver& Driver::Global() {
  static Driver driver;
  return driver;
}

Driver::Driver() {}

Driver::~Driver() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < drivers_.size(); i++) {
    void* library = drivers_[i].first;
    if (library) {
      dlclose(library);
    }
  }
  drivers_.clear();
}

size_t Driver::Count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return drivers_.size();
}

driver::Driver* Driver::At(int index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index >= 0 && index < drivers_.size()) {
    return drivers_[index].second;
  }
  return nullptr;
}

driver::Driver* Driver::Find(const char* name) {
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
