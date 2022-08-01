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
#include "core/types.h"
#include "driver/device.h"
#include "utility/micros.h"

namespace nnadapter {
namespace runtime {

class Device {
 public:
  explicit Device(const std::string& name);
  ~Device();

  bool IsValid() const { return device_ != nullptr; }
  const char* GetName() const {
    return IsValid() ? device_->second->name : nullptr;
  }
  const char* GetVendor() const {
    return IsValid() ? device_->second->vendor : nullptr;
  }
  NNAdapterDeviceType GetType() const {
    return IsValid() ? device_->second->type : -1;
  }
  int32_t GetVersion() const {
    return IsValid() ? device_->second->version : -1;
  }
  int CreateContext(const char* properties,
                    int (*callback)(int event_id, void* user_data),
                    void** context);
  void DestroyContext(void* context);
  int ValidateProgram(void* context,
                      const core::Model* model,
                      bool* supported_operations);
  int CreateProgram(void* context,
                    core::Model* model,
                    core::Cache* cache,
                    void** program);
  void DestroyProgram(void* program);
  int ExecuteProgram(void* program,
                     uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments);

 private:
  std::pair<void*, driver::Device*>* device_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

class DeviceManager {
 public:
  static DeviceManager& get();
  DeviceManager();
  ~DeviceManager();
  size_t Count();
  std::pair<void*, driver::Device*>* At(int index);
  std::pair<void*, driver::Device*>* Find(const char* name);

 private:
  std::mutex mutex_;
  std::vector<std::pair<void*, std::pair<void*, driver::Device*>*>> devices_;
  DISALLOW_COPY_AND_ASSIGN(DeviceManager);
};

}  // namespace runtime
}  // namespace nnadapter
