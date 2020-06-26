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

#include <memory>                                 // std::unique_ptr
#include "lite/backends/xpu/xpu_header_sitter.h"  // xpu_free
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

struct XPUScratchPad {
  XPUScratchPad(void* addr, bool is_l3) : addr_(addr), is_l3_(is_l3) {}

  void* addr_;
  bool is_l3_;
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const {
    if (!sp->is_l3_) {
      xpu_free(sp->addr_);
    }
    delete sp;
  }
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static XPUScratchPadGuard MallocScratchPad(size_t size, bool use_l3 = true);
};

}  // namespace lite
}  // namespace paddle
