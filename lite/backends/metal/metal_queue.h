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
// #include <mutex>
#include <vector>

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include "lite/backends/metal/metal_device.h"

namespace paddle {
namespace lite {

class metal_queue {
 public:
#if defined(__OBJC__)
  metal_queue(const metal_device* device, id<MTLCommandQueue> queue);
#endif

  void wait_until_complete() const;
  void wait_until_dispatch() const;

#if defined(__OBJC__)
  id<MTLCommandBuffer> create_command_buffer() const;
#else
  void* create_command_buffer() const;
#endif

 private:
#if defined(__OBJC__)
  mutable std::vector<id<MTLCommandBuffer>> command_buffers_;
  id<MTLCommandQueue> metal_queue_;
#else
  mutable std::vector<void*> command_buffers_;
  void* metal_queue_;
#endif
  metal_device* mtl_device_;
  //  mutable std::recursive_mutex command_buffers_lock_;
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_QUEUE_H_
