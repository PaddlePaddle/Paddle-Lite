// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "adnn/core/types.h"

namespace adnn {
namespace runtime {

Device* open_device(const char* properties, const Callback* callback) {
  return reinterpret_cast<Device*>(
      new adnn::runtime::Device(properties, callback));
}

void close_device(Device* device) {
  if (device) {
    delete reinterpret_cast<nnadapter::runtime::Device*>(device);
  }
}

Context* create_context(Device* device, const char* properties) {
  return reinterpret_cast<Context*>(new adnn::runtime::Context(
      reinterpret_cast<nnadapter::runtime::Device*>(device), properties));
}

void destroy_context(Context* context) {
  if (context) {
    delete reinterpret_cast<nnadapter::runtime::Context*>(device);
  }
}

}  // namespace runtime
}  // namespace adnn
