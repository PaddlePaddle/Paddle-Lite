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

#include "runtime/context.h"
#include <string>
#include "utility/logging.h"

namespace nnadapter {
namespace runtime {

Context::Context(std::vector<Device*> devices,
                 const std::string& properties,
                 int (*callback)(int event_id, void* user_data))
    : properties_(properties) {
  for (size_t i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    void* context = nullptr;
    NNADAPTER_CHECK_EQ(
        device->CreateContext(properties_.c_str(), callback, &context),
        NNADAPTER_NO_ERROR);
    device_contexts_.push_back({context, device});
  }
}

Context::~Context() {
  for (size_t i = 0; i < device_contexts_.size(); i++) {
    auto device_context = &device_contexts_[i];
    device_context->device->DestroyContext(device_context->context);
  }
}

Context::DeviceContext* Context::GetDeviceContext(const char* name) {
  for (size_t i = 0; i < device_contexts_.size(); i++) {
    auto device_context = &device_contexts_[i];
    if (device_context->device->IsValid()) {
      if (!strcmp(device_context->device->GetName(), name)) {
        return device_context;
      }
    } else {
      NNADAPTER_LOG(WARNING) << "Driver for device '" << name << "' not found.";
    }
  }
  return nullptr;
}

Context::DeviceContext* Context::GetDeviceContext(int index) {
  NNADAPTER_CHECK_GE(index, 0);
  NNADAPTER_CHECK_LT(index, device_contexts_.size())
      << "No device found, expected index < " << device_contexts_.size()
      << " but recevied " << index;
  auto device_context = &device_contexts_[index];
  NNADAPTER_CHECK(device_context->device->IsValid())
      << "Driver for device '" << device_context->device->GetName()
      << "' not found.";
  return device_context;
}

}  // namespace runtime
}  // namespace nnadapter
