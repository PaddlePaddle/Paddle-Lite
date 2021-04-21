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

#pragma once

#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>
#include "nnadapter_driver.h"  // NOLINT

namespace nnadapter {
namespace runtime {

class Device {
 public:
  explicit Device(const std::string& name);
  ~Device();

  bool hasDriver() const { return driver_ != nullptr; }
  const char* getName() const { return hasDriver() ? driver_->name : nullptr; }
  const char* getVendor() const {
    return hasDriver() ? driver_->vendor : nullptr;
  }
  NNAdapterDeviceType getType() const {
    return hasDriver() ? driver_->type : -1;
  }
  int32_t getVersion() const { return hasDriver() ? driver_->version : -1; }
  int createModelFromGraph(driver::Graph* graph, void** model);
  int createModelFromCache(void* buffer, size_t length, void** model);
  void destroyModel(void* model);
  int runModelSync(void* model,
                   uint32_t inputCount,
                   driver::Operand** inputs,
                   uint32_t outputCount,
                   driver::Operand** outputs);
  int runModelAsync(void* model,
                    uint32_t inputCount,
                    driver::Operand** inputs,
                    uint32_t outputCount,
                    driver::Operand** outputs);

 private:
  void* context_{nullptr};
  driver::Driver* driver_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

class Graph {
 public:
  Graph() : completed_{false} {}
  ~Graph();
  int addOperand(const NNAdapterOperandType& type, driver::Operand** operand);
  int addOperation(NNAdapterOperationType type, driver::Operation** operation);
  int identifyInputsAndOutputs(uint32_t inputCount,
                               driver::Operand** inputs,
                               uint32_t outputCount,
                               driver::Operand** outputs);
  int finish();

  driver::Graph graph_;
  bool completed_;
};

class Event {
 public:
  ~Event() = default;
  void wait() {}
  int getStatus() { return NNADAPTER_NO_ERROR; }
};

class Model {
 public:
  Model(Graph* graph, std::vector<Device*> devices)
      : graph_(graph),
        model_(nullptr),
        cached_(false),
        buffer_(nullptr),
        length_(0),
        devices_(devices),
        completed_{false} {}
  Model(void* buffer,
        size_t length,
        uint32_t inputCount,
        const NNAdapterOperandType** inputTypes,
        uint32_t outputCount,
        const NNAdapterOperandType** outputTypes,
        std::vector<Device*> devices);
  ~Model();
  Device* firstDevice();
  int setCaching(const char* cacheDir, const uint8_t* token);
  int finish();
  int setInput(int32_t index,
               const uint32_t* dimensions,
               uint32_t dimensionCount,
               void* buffer,
               size_t length);
  int setOutput(int32_t index,
                const uint32_t* dimensions,
                uint32_t dimensionCount,
                void* buffer,
                size_t length);
  int run(Event** event);

  Graph* graph_;
  void* model_;
  bool cached_;
  void* buffer_;
  size_t length_;
  std::vector<Device*> devices_;
  bool completed_;
};

class Execution {
 public:
  explicit Execution(Model* model) : model_(model) {}
  int setInput(int32_t index,
               const uint32_t* dimensions,
               uint32_t dimensionCount,
               void* buffer,
               size_t length);
  int setOutput(int32_t index,
                const uint32_t* dimensions,
                uint32_t dimensionCount,
                void* buffer,
                size_t length);
  int run(Event** event);

  Model* model_;
};

class Driver {
 public:
  static Driver& Global();
  Driver();
  ~Driver();
  size_t Count();
  driver::Driver* At(int index);
  driver::Driver* Find(const char* name);

 private:
  std::mutex mutex_;
  std::vector<std::pair<void*, driver::Driver*>> drivers_;
  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;
};

}  // namespace runtime
}  // namespace nnadapter
