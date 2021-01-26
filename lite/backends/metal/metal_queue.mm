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


#include "lite/core/dim.h"
#include "lite/utils/cp_logging.h"
#include "lite/backends/metal/metal_queue.h"


namespace paddle {
namespace lite {

MetalQueue::MetalQueue(const MetalDevice* device, id<MTLCommandQueue> queue) : metal_queue_(queue) {
  mtl_device_ = const_cast<MetalDevice*>(device);
}

id<MTLCommandBuffer> MetalQueue::CreateCommandBuffer() const {
  id<MTLCommandBuffer> cmd_buffer = [metal_queue_ commandBufferWithUnretainedReferences];
  [cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    //    std::lock_guard<std::recursive_mutex> lock(command_buffers_lock_);
    const auto iter = find_if(
        command_buffers_.cbegin(),
        command_buffers_.cend(),
        [&buffer](const decltype(command_buffers_)::value_type& elem) { return (elem == buffer); });
    if (iter == command_buffers_.cend()) {
      LOG(ERROR) << "failed to create command buffer"
                 << "\n";
      return;
    }
    command_buffers_.erase(iter);
  }];
  {
    // std::lock_guard<std::recursive_mutex> lock(command_buffers_lock_);
    command_buffers_.emplace_back(cmd_buffer);
  }
  return cmd_buffer;
}

void MetalQueue::WaitUntilComplete() const {
  decltype(command_buffers_) cur_cmd_buffers;
  {
    // std::lock_guard<std::recursive_mutex> lock(command_buffers_lock_);
    cur_cmd_buffers = command_buffers_;
  }
  for (const auto& cmd_buffer : cur_cmd_buffers) {
    [cmd_buffer waitUntilCompleted];
  }
}

void MetalQueue::WaitUntilDispatch() const {
  decltype(command_buffers_) cur_cmd_buffers;
  {
    // std::lock_guard<std::recursive_mutex> lock(command_buffers_lock_);
    cur_cmd_buffers = command_buffers_;
  }
  for (const auto& cmd_buffer : cur_cmd_buffers) {
    [cmd_buffer waitUntilScheduled];
  }
}

}
}
