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

  bool IsValid() const { return device_ != nullptr; }
  const char* GetName() const { return IsValid() ? device_->name : nullptr; }
  const char* GetVendor() const {
    return IsValid() ? device_->vendor : nullptr;
  }
  NNAdapterDeviceType GetType() const { return IsValid() ? device_->type : -1; }
  int32_t GetVersion() const { return IsValid() ? device_->version : -1; }
  int CreateContext(void** context);
  void DestroyContext(void* context);
  int CreateProgram(void* context,
                    driver::Model* model,
                    driver::Cache* cache,
                    void** program);
  void DestroyProgram(void* program);
  int ExecuteProgram(void* program,
                     uint32_t input_count,
                     driver::Argument* input_arguments,
                     uint32_t output_count,
                     driver::Argument* output_arguments);

 private:
  driver::Device* device_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

class Context {
 public:
  explicit Context(std::vector<Device*> devices);
  ~Context();
  std::pair<void*, Device*> GetFirstDevice();

 private:
  std::vector<std::pair<void*, Device*>> contexts_;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

class Model {
 public:
  Model() : completed_{false} {}
  ~Model();
  int AddOperand(const NNAdapterOperandType& type, driver::Operand** operand);
  int AddOperation(NNAdapterOperationType type, driver::Operation** operation);
  int IdentifyInputsAndOutputs(uint32_t input_count,
                               driver::Operand** input_operands,
                               uint32_t output_count,
                               driver::Operand** output_operands);
  int Finish();

  driver::Model model_;
  bool completed_;
};

class Compilation {
 public:
  Compilation(Model* model,
              const char* cache_key,
              void* cache_buffer,
              uint32_t cache_length,
              const char* cache_dir,
              Context* context);
  ~Compilation();
  int Finish();
  int QueryInputsAndOutputs(uint32_t* input_count,
                            NNAdapterOperandType** input_types,
                            uint32_t* output_count,
                            NNAdapterOperandType** output_types);
  int Execute(std::vector<driver::Argument>* input_arguments,
              std::vector<driver::Argument>* output_arguments);

 private:
  Model* model_{nullptr};
  driver::Cache cache_;
  Context* context_{nullptr};
  void* program_{nullptr};
  bool completed_{false};
};

class Execution {
 public:
  explicit Execution(Compilation* compilation) : compilation_(compilation) {}
  int SetInput(int32_t index,
               const int32_t* dimensions,
               uint32_t dimension_count,
               void* buffer,
               uint32_t length);
  int SetOutput(int32_t index,
                const int32_t* dimensions,
                uint32_t dimensionCount,
                void* buffer,
                uint32_t length);
  int Compute();

 private:
  Compilation* compilation_{nullptr};
  std::vector<driver::Argument> input_arguments_;
  std::vector<driver::Argument> output_arguments_;
};

class DeviceManager {
 public:
  static DeviceManager& Global();
  DeviceManager();
  ~DeviceManager();
  size_t Count();
  driver::Device* At(int index);
  driver::Device* Find(const char* name);

 private:
  std::mutex mutex_;
  std::vector<std::pair<void*, driver::Device*>> devices_;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;
};

}  // namespace runtime
}  // namespace nnadapter
