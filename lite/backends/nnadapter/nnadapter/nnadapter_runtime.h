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

class Network {
 public:
  Network() {}

 private:
  driver::Network* network{nullptr};
};

class Model {
 public:
  Model() {}

 private:
  void* model{nullptr};
};

class Execution {
 public:
  Execution() {}

 private:
};

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
  int32_t buildModel(driver::Network* network, void** model);
  int32_t excuteModel(void* model);

 private:
  void* context_{nullptr};
  driver::Driver* driver_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
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
