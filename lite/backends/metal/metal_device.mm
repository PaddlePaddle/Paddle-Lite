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


#include "lite/backends/metal/metal_device.h"
#include "lite/utils/cp_logging.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {

std::shared_ptr<MetalQueue> MetalDevice::CreateQueue() const {
  id<MTLCommandQueue> queue = [device_ newCommandQueue];
  if (queue == nil) {
    LOG(ERROR) << "ERROR: fail to create command queue"
               << "\n";
    return {};
  }

  auto ret = std::make_shared<MetalQueue>(this, queue);
  queues_.push_back(ret);
  return ret;
}

std::shared_ptr<MetalQueue> MetalDevice::GetDefaultQueue() const {
  if (queues_.size() > 0) {
    return queues_[0];
  } else {
    return CreateQueue();
  }
}

MetalDevice::~MetalDevice(){
#if (!__has_feature(objc_arc))
    for(auto item : queues_) {
      [item->queue() release];
      queues_.pop_back();
    }
    queues_.clear();
#endif
}

id<MTLDevice> MetalDevice::device() const { return device_; }
void MetalDevice::set_device(id<MTLDevice> device) { device_ = device; }
void MetalDevice::set_context(MetalContext *context) { context_ = context; }
void MetalDevice::set_name(const char *name) { name_ = name; }
}
}