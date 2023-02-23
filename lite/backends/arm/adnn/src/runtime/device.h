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

namespace adnn {
namespace runtime {

class Device {
 public:
  explicit Device(const char* properties, const Callback* callback);
  ~Device();
  void* CreateContext(const char* properties);
  void DestroyContext(void* context);
  void* Alloc(size_t size);
  void Free(void* ptr);
  void* AlignedAlloc(size_t alignment, size_t size);
  void AlignedFree(void* context, void* ptr);

 private:
  const Callback* callback_{nullptr};
  void* device_{nullptr};
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

}  // namespace runtime
}  // namespace adnn
