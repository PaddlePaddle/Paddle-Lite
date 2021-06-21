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
#include "utility/logging.h"

namespace nnadapter {
namespace runtime {

Context::Context(std::vector<Device*> devices) {
  for (size_t i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    void* context = nullptr;
    device->CreateContext(&context);
    contexts_.emplace_back(context, device);
  }
}

Context::~Context() {
  for (size_t i = 0; i < contexts_.size(); i++) {
    auto device = contexts_[i].second;
    void* context = contexts_[i].first;
    device->DestroyContext(context);
  }
}

std::pair<void*, Device*> Context::GetFirstDevice() {
  NNADAPTER_CHECK_GT(contexts_.size(), 0) << "No device found.";
  auto first_device = contexts_[0];
  NNADAPTER_CHECK(first_device.second->IsValid())
      << "Driver for device '" << first_device.second->GetName()
      << "' not found.";
  return first_device;
}

}  // namespace runtime
}  // namespace nnadapter
