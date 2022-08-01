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

#include <memory>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/utils/log/cp_logging.h"

#define XPU_CALL(func)                                        \
  {                                                           \
    auto e = (func);                                          \
    CHECK_EQ(e, 0) << "XPU: (" << #func << ") returns " << e; \
  }

namespace paddle {
namespace lite {

struct XPUScratchPad {
  XPUScratchPad(void* addr, size_t size) : addr_(addr), size_(size) {}
  // XXX(miaotianxiang): |size_| increases monotonically
  void Reserve(size_t new_size);
  void* addr_{nullptr};
  size_t size_{0};
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const;
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

class XPUMemory {
 public:
  static void* Malloc(size_t size);
  static void Free(void* ptr);
  static void MemcpyHtoDSync(void* dst, const void* src, size_t size);
  static void MemcpyDtoHSync(void* dst, const void* src, size_t size);
  static XPUScratchPadGuard MallocScratchPad(size_t size);
  static int get_max_ptr_size();
};

}  // namespace lite
}  // namespace paddle
