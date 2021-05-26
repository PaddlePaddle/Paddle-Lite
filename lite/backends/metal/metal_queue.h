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

class MetalKernel;
class MetalKernelProgram;
class RuntimeProgram;

struct MetalCommandBuffer {
#if defined(__OBJC__)
  id<MTLCommandBuffer> metal_command_buffer_{nil};
#else
  void* metal_command_buffer_{nullptr};
#endif
  bool have_command_{false};

  virtual ~MetalCommandBuffer();
};

struct MetalEncoder {
  MetalEncoder(MetalCommandBuffer* buffer, MetalKernelProgram* program);
  virtual ~MetalEncoder();

#if defined(__OBJC__)
  id<MTLCommandBuffer> metal_command_buffer_{nil};
  id<MTLComputeCommandEncoder> metal_command_encoder_{nil};
#else
  void* metal_command_buffer_{nullptr};
  void* metal_command_encoder_{nullptr};
#endif
};

class MetalQueue {
 public:
#if defined(__OBJC__)
  MetalQueue(const MetalDevice* device, id<MTLCommandQueue> queue);
#endif

  std::unique_ptr<MetalCommandBuffer> CreateCommandBuffer(
      RuntimeProgram* program);

#if defined(__OBJC__)
  id<MTLCommandQueue> queue() { return queue_; }
#else
  void* queue() { return queue_; }
#endif

 private:
#if defined(__OBJC__)
  mutable std::vector<id<MTLCommandBuffer>> command_buffers_;
  id<MTLCommandQueue> queue_;
#else
  mutable std::vector<void*> command_buffers_;
  void* queue_;
#endif
  MetalDevice* mtl_device_;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_QUEUE_H_
