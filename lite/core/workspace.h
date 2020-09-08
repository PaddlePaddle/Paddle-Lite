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
#include "lite/core/memory.h"
#include "lite/core/types.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

/*
 * WorkSpace is a container that help to manage the temporary memory that are
 * shared across kernels during the serial execution.
 *
 * Due to the mobile library size limit, a complex allocator or GC algorithm is
 * not suitable here, one need to carefully manage the workspace inside a single
 * kernel.
 *
 * NOTE
 *
 * For kernel developers, one need to call the workspace as follows:
 *
 * - call `WorkSpace::Global().Alloc()` if needed to allocate some temporary
 * buffer.
 */
class WorkSpace {
 public:
  // Reset the workspace, and treat the workspace as empty.
  void AllocReset() { cursor_ = 0; }

  // Allocate a memory buffer.
  core::byte_t* Alloc(size_t size) {
    buffer_.ResetLazy(target_, cursor_ + size);
    auto* data = static_cast<core::byte_t*>(buffer_.data()) + cursor_;
    cursor_ += size;
    return data;
  }

  static WorkSpace& Global_Host() {
    static LITE_THREAD_LOCAL std::unique_ptr<WorkSpace> x(
        new WorkSpace(TARGET(kHost)));
    return *x;
  }

#if defined(LITE_WITH_X86)
  static WorkSpace& Global_X86() { return Global_Host(); }
#endif

#if defined(LITE_WITH_ARM)
  static WorkSpace& Global_ARM() { return Global_Host(); }
#endif

#if defined(LITE_WITH_CUDA)
  static WorkSpace& Global_CUDA() {
    static LITE_THREAD_LOCAL std::unique_ptr<WorkSpace> x(
        new WorkSpace(TARGET(kCUDA)));
    return *x;
  }
#endif

#if defined(LITE_WITH_MLU)
  static WorkSpace& Global_MLU() {
    static LITE_THREAD_LOCAL std::unique_ptr<WorkSpace> x(
        new WorkSpace(TARGET(kMLU)));
    return *x;
  }
#endif

 private:
  explicit WorkSpace(TargetType x) : target_(x) {}

  TargetType target_;
  Buffer buffer_;
  size_t cursor_;

  DISALLOW_COPY_AND_ASSIGN(WorkSpace);
};

}  // namespace lite
}  // namespace paddle
