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

#include "lite/backends/metal/metal_queue.h"
#include "lite/backends/metal/metal_kernel.h"
#include "lite/core/dim.h"
#include "lite/core/program.h"

//#define LITE_METAL_SAVE_TENSOR 1

namespace paddle {
namespace lite {

MetalQueue::MetalQueue(const MetalDevice* device, id<MTLCommandQueue> queue)
    : queue_(queue) {
  mtl_device_ = const_cast<MetalDevice*>(device);
}

std::unique_ptr<MetalCommandBuffer> MetalQueue::CreateCommandBuffer(RuntimeProgram* program) {
  id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
  [cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
#if LITE_METAL_SAVE_TENSOR
  if( program != nullptr ) {
    program->SaveOutput();
  }
#endif
  }];
  auto cmd_buf = new MetalCommandBuffer();
  std::unique_ptr<MetalCommandBuffer> ret(cmd_buf);
  ret->metal_command_buffer_ = cmd_buffer;
  return ret;
}


MetalEncoder::MetalEncoder(MetalCommandBuffer* buffer, MetalKernelProgram* program){
    metal_command_buffer_ = buffer->metal_command_buffer_;
    metal_command_encoder_ = [buffer->metal_command_buffer_ computeCommandEncoder];
    [metal_command_encoder_ setComputePipelineState:(program->pipeline_state_)];
    buffer->have_command_ = true;
}

MetalEncoder::~MetalEncoder() {
  metal_command_buffer_ = nil;
}

MetalCommandBuffer::~MetalCommandBuffer() {
    metal_command_buffer_ = nil;
    have_command_ = false;
}

}
}
