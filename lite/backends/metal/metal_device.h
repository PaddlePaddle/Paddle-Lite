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

#ifndef LITE_BACKENDS_METAL_METAL_DEVICE_H_
#define LITE_BACKENDS_METAL_METAL_DEVICE_H_

#if defined(__OBJC__)
#import <Metal/Metal.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace lite {

class MetalQueue;
class MetalContext;

class MetalDevice {
 public:
  std::shared_ptr<MetalQueue> CreateQueue() const;
  std::shared_ptr<MetalQueue> GetDefaultQueue() const;

#if defined(__OBJC__)
  id<MTLDevice> device() const;
  void set_device(id<MTLDevice> device);
#else
  void *device() const;
  void set_device(void *device);
#endif

  void set_context(MetalContext *context);
  void set_name(const char *name);
  MetalContext *context() { return context_; }
  std::string name() { return name_; }
  virtual ~MetalDevice();

 private:
#if defined(__OBJC__)
  id<MTLDevice> device_;
#else
  void *device_;
#endif

  MetalContext *context_;
  mutable std::vector<std::shared_ptr<MetalQueue>> queues_;
  std::string name_;
};
}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_DEVICE_H_
