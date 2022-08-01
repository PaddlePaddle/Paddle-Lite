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

#include <string>
#include <utility>
#include <vector>
#include "runtime/device.h"

namespace nnadapter {
namespace runtime {

class Context {
 public:
  typedef struct {
    void* context;
    Device* device;
  } DeviceContext;
  explicit Context(std::vector<Device*> devices,
                   const std::string& properties,
                   int (*callback)(int event_id, void* user_data));
  ~Context();
  DeviceContext* GetDeviceContext(const char* name);
  DeviceContext* GetDeviceContext(int index);
  size_t GetDeviceCount() { return device_contexts_.size(); }
  const char* GetProperties() { return properties_.c_str(); }

 private:
  std::vector<DeviceContext> device_contexts_;
  std::string properties_;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

}  // namespace runtime
}  // namespace nnadapter
