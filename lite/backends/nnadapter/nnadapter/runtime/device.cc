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

#include "runtime/device.h"
#include <dlfcn.h>
#include <stdlib.h>
#include "utility/logging.h"
#include "utility/micros.h"

namespace nnadapter {
namespace runtime {

Device::Device(const std::string& name) {
  device_ = DeviceManager::Global().Find(name.c_str());
}

Device::~Device() { device_ = nullptr; }

int Device::CreateContext(void** context) {
  if (device_ && context) {
    return device_->create_context(context);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

void Device::DestroyContext(void* context) {
  if (device_ && context) {
    device_->destroy_context(context);
  }
}

int Device::CreateProgram(void* context,
                          hal::Model* model,
                          hal::Cache* cache,
                          void** program) {
  if (device_ && context && model && program) {
    return device_->create_program(context, model, cache, program);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

void Device::DestroyProgram(void* program) {
  if (device_ && program) {
    return device_->destroy_program(program);
  }
}

int Device::ExecuteProgram(void* program,
                           uint32_t input_count,
                           hal::Argument* input_arguments,
                           uint32_t output_count,
                           hal::Argument* output_arguments) {
  if (device_ && program && output_arguments && output_count) {
    return device_->execute_program(
        program, input_count, input_arguments, output_count, output_arguments);
  }
  return NNADAPTER_INVALID_PARAMETER;
}

DeviceManager& DeviceManager::Global() {
  static DeviceManager manager;
  return manager;
}

DeviceManager::DeviceManager() {}

DeviceManager::~DeviceManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < devices_.size(); i++) {
    void* library = devices_[i].first;
    if (library) {
      dlclose(library);
    }
  }
  devices_.clear();
}

size_t DeviceManager::Count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_.size();
}

hal::Device* DeviceManager::At(int index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index >= 0 && index < devices_.size()) {
    return devices_[index].second;
  }
  return nullptr;
}

hal::Device* DeviceManager::Find(const char* name) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < devices_.size(); i++) {
    auto device = devices_[i].second;
    if (strcmp(device->name, name) == 0) {
      return device;
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
  auto device = reinterpret_cast<hal::Device*>(dlsym(library, symbol.c_str()));
  if (!device) {
    dlclose(library);
    NNADAPTER_LOG(ERROR) << "Failed to find the symbol '" << symbol << "' from "
                         << path << ", " << dlerror();
    return nullptr;
  }
  devices_.emplace_back(library, device);
  return device;
}

}  // namespace runtime
}  // namespace nnadapter
