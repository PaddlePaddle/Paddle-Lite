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
#include "context.h"                  // NOLINT
#include "program.h"                  // NOLINT

namespace nnadapter {
namespace driver {
namespace rockchip_npu {

int CreateContext(void** context) {
  if (!context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = new Context();
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(ERROR) << "Failed to create context for rockchip_npu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<void*>(c);
  return NNADAPTER_NO_ERROR;
}

void DestroyContext(void* context) {
  if (!context) {
    auto c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int CreateProgram(void* context,
                  driver::Model* model,
                  driver::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for rockchip_npu.";
  if (!context || !(model && cache) || !program) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *program = nullptr;
  auto p = new Program();
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Build(model, cache);
  if (result == NNADAPTER_NO_ERROR) {
    *program = reinterpret_cast<void*>(p);
  }
  return result;
}

void DestroyProgram(void* context, void* program) {
  if (context && program) {
    NNADAPTER_LOG(INFO) << "Destroy program for rockchip_npu.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* context,
                   void* program,
                   uint32_t input_count,
                   driver::Argument* input_arguments,
                   uint32_t output_count,
                   driver::Argument* output_arguments) {
  if (!context || !program || !output_arguments || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
  return p->Execute(
      input_count, input_arguments, output_count, output_arguments);
}

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter

nnadapter::driver::Driver NNADAPTER_EXPORT
    NNADAPTER_AS_SYM2(NNADAPTER_DRIVER_TARGET) = {
        .name = NNADAPTER_AS_STR2(NNADAPTER_DRIVER_NAME),
        .vendor = "Rockchip",
        .type = NNADAPTER_ACCELERATOR,
        .version = 1,
        .create_context = nnadapter::driver::rockchip_npu::CreateContext,
        .destroy_context = nnadapter::driver::rockchip_npu::DestroyContext,
        .create_program = nnadapter::driver::rockchip_npu::CreateProgram,
        .destroy_program = nnadapter::driver::rockchip_npu::DestroyProgram,
        .execute_program = nnadapter::driver::rockchip_npu::ExecuteProgram,
};
