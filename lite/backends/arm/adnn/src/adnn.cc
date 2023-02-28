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

#include "runtime/context.h"
#include "runtime/device.h"
#include "utilities/dll_export.h"

namespace adnn {

ADNN_DLL_EXPORT Device* open_device(int thread_num, const Callback* callback) {
  return reinterpret_cast<Device*>(new runtime::Device(thread_num, callback));
}

ADNN_DLL_EXPORT void close_device(Device* device) {
  if (device) {
    delete reinterpret_cast<runtime::Device*>(device);
  }
}

ADNN_DLL_EXPORT Context* create_context(Device* device, int thread_num) {
  return reinterpret_cast<Context*>(new runtime::Context(
      reinterpret_cast<runtime::Device*>(device), thread_num));
}

ADNN_DLL_EXPORT void destroy_context(Context* context) {
  if (context) {
    delete reinterpret_cast<runtime::Context*>(context);
  }
}

}  // namespace adnn
