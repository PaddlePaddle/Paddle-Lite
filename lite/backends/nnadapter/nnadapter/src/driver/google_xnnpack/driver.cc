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

#include "driver/device.h"
#include "driver/google_xnnpack/engine.h"
#include "utility/logging.h"
#include "utility/micros.h"

namespace nnadapter {
namespace google_xnnpack {

int OpenDevice(void** device) {
  auto d = new Device();
  if (!d) {
    *device = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to open device for google_xnnpack.";
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

int CreateContext(void* device,
                  const char* properties,
                  int (*callback)(int event_id, void* user_data),
                  void** context) {
  if (!device || !context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<Device*>(device);
  auto c = new Context(d, properties);
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to create context for google_xnnpack.";
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

int ValidateProgram(void* context,
                    const core::Model* model,
                    bool* supported_operations) {
  NNADAPTER_LOG(INFO) << "Validate program for google_xnnpack.";
  if (!context || !model || !supported_operations) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = reinterpret_cast<Context*>(context);
  auto p = new Program(c);
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Validate(model, supported_operations);
  delete p;
  return result;
}

int CreateProgram(void* context,
                  core::Model* model,
                  core::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for google_xnnpack.";
  if (!context || !(model || (cache && cache->buffer.size())) || !program) {
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
    NNADAPTER_LOG(INFO) << "Destroy program for google_xnnpack.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* program,
                   uint32_t input_count,
                   core::Argument* input_arguments,
                   uint32_t output_count,
                   core::Argument* output_arguments) {
  if (!program || !output_arguments || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
  return p->Execute(
      input_count, input_arguments, output_count, output_arguments);
}

}  // namespace google_xnnpack
}  // namespace nnadapter

NNADAPTER_EXPORT nnadapter::driver::Device NNADAPTER_AS_SYM2(DEVICE_NAME) = {
    .name = NNADAPTER_AS_STR2(DEVICE_NAME),
    .vendor = "Google",
    .type = NNADAPTER_ACCELERATOR,
    .version = 1,
    .open_device = nnadapter::google_xnnpack::OpenDevice,
    .close_device = nnadapter::google_xnnpack::CloseDevice,
    .create_context = nnadapter::google_xnnpack::CreateContext,
    .destroy_context = nnadapter::google_xnnpack::DestroyContext,
    .validate_program = nnadapter::google_xnnpack::ValidateProgram,
    .create_program = nnadapter::google_xnnpack::CreateProgram,
    .destroy_program = nnadapter::google_xnnpack::DestroyProgram,
    .execute_program = nnadapter::google_xnnpack::ExecuteProgram,
};
