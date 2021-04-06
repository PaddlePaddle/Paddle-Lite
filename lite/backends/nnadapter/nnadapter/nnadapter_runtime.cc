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

int32_t Device::buildModel(driver::Network* network, void** model) {
  if (driver_ && context_) {
    return driver_->buildModel(network, context_, model);
  }
  return NNADAPTER_INVALID_OBJECT;
}

int32_t Device::excuteModel(void* model) {
  if (driver_ && context_) {
    return driver_->excuteModel(context_, model);
  }
  return NNADAPTER_INVALID_OBJECT;
}

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
    driver::Driver* driver = drivers_[i].second;
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
  driver::Driver* driver = (driver::Driver*)dlsym(library, symbol.c_str());
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
