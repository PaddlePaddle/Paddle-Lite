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

#include <string>
#include <vector>

#include "lite/backends/metal/metal_context.h"
#include "lite/core/dim.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

using TargetWrapperMetal = TargetWrapper<TARGET(kMetal)>;

template <>
class TargetWrapper<TARGET(kMetal)> {
 public:
  static size_t num_devices();
  static size_t GetCurDevice() {
    int dev_id = 0;
    return dev_id;
  }

  static bool MPSVersionRequired();

  static void CreateCommandBuffer(RuntimeProgram* program) {
    assert(program);
    ctx_.CreateCommandBuffer(program);
    return;
  }

  static void WaitForCompleted();

  static void set_metal_path(std::string path) { ctx_.set_metal_path(path); }

  static void set_metal_use_aggressive_optimization(bool flag) {
    ctx_.set_use_aggressive_optimization(flag);
  }

  static bool use_mps() { return ctx_.use_mps(); }

  static bool use_aggressive_optimization() {
    return ctx_.use_aggressive_optimization();
  }

  static void set_metal_use_mps(bool flag) { ctx_.set_use_mps(flag); }

  template <typename T>
  static void* MallocImage(const DDim dim,
                           std::vector<int> transport,
                           void* host_ptr = nullptr);

  template <typename T>
  static void* MallocBuffer(const DDim dim,
                            bool transpose,
                            bool to_nhwc,
                            bool pad_when_one_c,
                            void* host_ptr);

  static void FreeImage(void* image);

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static void MemsetSync(void* devPtr, int value, size_t count);

  static void* Map(void* data, int offset, size_t size);
  static void UnMap(void* data);

  static LITE_THREAD_LOCAL MetalContext ctx_;
};
}  // namespace lite
}  // namespace paddle
