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

#include "driver.h"                   // NOLINT
#include "../../nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {
namespace rockchip_npu {

int32_t createContext(void** context) {
  if (!context) {
    return NNADAPTER_INVALID_OBJECT;
  }
  Context* c = new Context(nullptr);
  if (c == nullptr) {
    *context = nullptr;
    NNADAPTER_LOG(ERROR) << "Failed to create context for rockchip_npu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<void*>(c);
  return NNADAPTER_NO_ERROR;
}

void destroyContext(void* context) {
  if (!context) {
    Context* c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int32_t buildModel(Network* network, void* context, void** model) {
  *model = nullptr;
  return VerifyNetwork(network);
}

int32_t excuteModel(void* context, void* model) { return NNADAPTER_NO_ERROR; }

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter

nnadapter::driver::Driver NNADAPTER_EXPORT
    NNADAPTER_AS_SYM2(NNADAPTER_DRIVER_TARGET) = {
        .name = NNADAPTER_AS_STR2(NNADAPTER_DRIVER_NAME),
        .vendor = "Rockchip",
        .type = NNADAPTER_ACCELERATOR,
        .version = 1,
        .createContext = nnadapter::driver::rockchip_npu::createContext,
        .destroyContext = nnadapter::driver::rockchip_npu::destroyContext,
        .buildModel = nnadapter::driver::rockchip_npu::buildModel,
        .excuteModel = nnadapter::driver::rockchip_npu::excuteModel};
