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

#include "driver/amlogic_npu/converter.h"
#include "utility/logging.h"
#include "utility/micros.h"

namespace nnadapter {
namespace amlogic_npu {

int OpenDevice(void** device) {
  auto d = new Device();
  if (!d) {
    *device = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to open device for amlogic_npu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *device = reinterpret_cast<void*>(d);
  return NNADAPTER_NO_ERROR;
}

void CloseDevice(void* device) {
  if (device) {
    auto d = reinterpret_cast<Device*>(device);
    delete d;
  }
}

int CreateContext(void* device, const char* properties, void** context) {
  if (!device || !context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<Device*>(device);
  auto c = new Context(d, properties);
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to create context for amlogic_npu.";
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
                  hal::Model* model,
                  hal::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for amlogic_npu.";
  if (!context || !(model && cache) || !program) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *program = nullptr;
  auto c = reinterpret_cast<Context*>(context);
  auto p = new Program(c);
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Build(model, cache);
  if (result == NNADAPTER_NO_ERROR) {
    *program = reinterpret_cast<void*>(p);
  }
  return result;
}

void DestroyProgram(void* program) {
  if (program) {
    NNADAPTER_LOG(INFO) << "Destroy program for amlogic_npu.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* program,
                   uint32_t input_count,
                   hal::Argument* input_arguments,
                   uint32_t output_count,
                   hal::Argument* output_arguments) {
  if (!program || !output_arguments || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
  return p->Execute(
      input_count, input_arguments, output_count, output_arguments);
}

}  // namespace amlogic_npu
}  // namespace nnadapter

NNADAPTER_EXPORT nnadapter::hal::Device NNADAPTER_AS_SYM2(
    NNADAPTER_DRIVER_NAME) = {
    .name = NNADAPTER_AS_STR2(NNADAPTER_DEVICE_NAME),
    .vendor = "Amlogic",
    .type = NNADAPTER_ACCELERATOR,
    .version = 1,
    .open_device = nnadapter::amlogic_npu::OpenDevice,
    .close_device = nnadapter::amlogic_npu::CloseDevice,
    .create_context = nnadapter::amlogic_npu::CreateContext,
    .destroy_context = nnadapter::amlogic_npu::DestroyContext,
    .create_program = nnadapter::amlogic_npu::CreateProgram,
    .destroy_program = nnadapter::amlogic_npu::DestroyProgram,
    .execute_program = nnadapter::amlogic_npu::ExecuteProgram,
};
