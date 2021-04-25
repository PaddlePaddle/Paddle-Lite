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

  bool HasDriver() const { return driver_ != nullptr; }
  const char* GetName() const { return HasDriver() ? driver_->name : nullptr; }
  const char* GetVendor() const {
    return HasDriver() ? driver_->vendor : nullptr;
  }
  NNAdapterDeviceType GetType() const {
    return HasDriver() ? driver_->type : -1;
  }
  int32_t GetVersion() const { return HasDriver() ? driver_->version : -1; }
  int CreateProgram(driver::Model* model, driver::Cache* cache, void** program);
  void DestroyProgram(void* program);
  int ExecuteProgram(void* program,
                     uint32_t input_count,
                     driver::Argument* inputs,
                     uint32_t output_count,
                     driver::Argument* outputs);

 private:
  void* context_{nullptr};
  driver::Driver* driver_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

class Model {
 public:
  Model() : completed_{false} {}
  ~Model();
  int AddOperand(const NNAdapterOperandType& type, driver::Operand** operand);
  int AddOperation(NNAdapterOperationType type, driver::Operation** operation);
  int IdentifyInputsAndOutputs(uint32_t input_count,
                               driver::Operand** inputs,
                               uint32_t output_count,
                               driver::Operand** outputs);
  int Finish();

  driver::Model model_;
  bool completed_;
};

class Event {
 public:
  ~Event() = default;
  void Wait() {}
  int GetStatus() { return NNADAPTER_NO_ERROR; }
};

class Compilation {
 public:
  Compilation(Model* model,
              const char* cache_key,
              void* cache_buffer,
              size_t cache_length,
              const char* cache_dir,
              std::vector<Device*> devices);
  ~Compilation();
  Device* GetFirstDevice();
  int Finish();
  int Execute(std::vector<driver::Argument>* inputs,
              std::vector<driver::Argument>* outputs);

  Model* model_{nullptr};
  driver::Cache cache_;
  void* program_{nullptr};
  std::vector<Device*> devices_;
  bool completed_{false};
};

class Execution {
 public:
  explicit Execution(Compilation* compilation) : compilation_(compilation) {}
  int SetInput(int32_t index,
               const uint32_t* dimensions,
               uint32_t dimension_count,
               void* buffer,
               size_t length);
  int SetOutput(int32_t index,
                const uint32_t* dimensions,
                uint32_t dimensionCount,
                void* buffer,
                size_t length);
  int Compute();

 private:
  Compilation* compilation_{nullptr};
  std::vector<driver::Argument> inputs_;
  std::vector<driver::Argument> outputs_;
};

class DriverManager {
 public:
  static DriverManager& Global();
  DriverManager();
  ~DriverManager();
  size_t Count();
  driver::Driver* At(int index);
  driver::Driver* Find(const char* name);

 private:
  std::mutex mutex_;
  std::vector<std::pair<void*, driver::Driver*>> drivers_;
  DriverManager(const DriverManager&) = delete;
  DriverManager& operator=(const DriverManager&) = delete;
};

}  // namespace runtime
}  // namespace nnadapter
