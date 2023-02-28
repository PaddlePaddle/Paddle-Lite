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

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

namespace adnn {

/**
 * Result code.
 */
typedef enum {
  SUCCESS = 0,
  OUT_OF_MEMORY = 1,
  INVALID_PARAMETER = 2,
  FEATURE_NOT_SUPPORTED = 3
} Status;

typedef struct {
  void* (*open_device)(int thread_num);
  void (*close_device)(void* device);
  void* (*create_context)(void* device, int thread_num);
  void (*destroy_context)(void* context);
  void* (*alloc)(void* context, size_t size);
  void (*free)(void* context, void* ptr);
  void* (*aligned_alloc)(void* context, size_t alignment, size_t size);
  void (*aligned_free)(void* context, void* ptr);
} Callback;

typedef struct Device Device;
typedef struct Context Context;

}  // namespace adnn
