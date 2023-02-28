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

#include "adnn/core/types.h"
#include "runtime/device.h"

namespace adnn {
namespace runtime {

class Context {
 public:
  explicit Context(Device* device, int thread_num);
  ~Context();
  void* Alloc(size_t size);
  void Free(void* ptr);
  void* AlignedAlloc(size_t alignment, size_t size);
  void AlignedFree(void* ptr);
  int GetThreadNum() { return thread_num_; }

 private:
  Device* device_{nullptr};
  void* context_{nullptr};
  int thread_num_{1};
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

}  // namespace runtime
}  // namespace adnn
