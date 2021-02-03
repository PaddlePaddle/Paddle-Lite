// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_METAL_METAL_QUEUE_H_
#define LITE_BACKENDS_METAL_METAL_QUEUE_H_

#include <memory>
#include <vector>

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include "lite/backends/metal/metal_device.h"

namespace paddle {
namespace lite {

class MetalQueue {
 public:
#if defined(__OBJC__)
  MetalQueue(const MetalDevice* device, id<MTLCommandQueue> queue);
#endif

  void WaitUntilComplete() const;
  void WaitUntilDispatch() const;

#if defined(__OBJC__)
  id<MTLCommandBuffer> CreateCommandBuffer() const;
#else
  void* CreateCommandBuffer() const;
#endif

 private:
#if defined(__OBJC__)
  mutable std::vector<id<MTLCommandBuffer>> command_buffers_;
  id<MTLCommandQueue> metal_queue_;
#else
  mutable std::vector<void*> command_buffers_;
  void* metal_queue_;
#endif
  MetalDevice* mtl_device_;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_QUEUE_H_
