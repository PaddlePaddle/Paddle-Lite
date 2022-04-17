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

#include "core/types.h"

namespace nnadapter {
namespace driver {

typedef struct Device {
  // Properties
  const char* name;
  const char* vendor;
  NNAdapterDeviceType type;
  int32_t version;
  // Interfaces
  int (*open_device)(void** device);
  void (*close_device)(void* device);
  int (*create_context)(void* device,
                        const char* properties,
                        int (*callback)(int event_id, void* user_data),
                        void** context);
  void (*destroy_context)(void* context);
  int (*validate_program)(void* context,
                          const core::Model* model,
                          bool* supported_operations);
  int (*create_program)(void* context,
                        core::Model* model,
                        core::Cache* cache,
                        void** program);
  void (*destroy_program)(void* program);
  int (*execute_program)(void* program,
                         uint32_t input_count,
                         core::Argument* input_arguments,
                         uint32_t output_count,
                         core::Argument* output_arguments);
} Device;

}  // namespace driver
}  // namespace nnadapter
