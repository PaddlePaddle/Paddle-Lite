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

//
// Created by liuzheyuan on 2020/9/30.
//

#ifndef LITE_BACKENDS_METAL_METAL_DEVICE_H_
#define LITE_BACKENDS_METAL_METAL_DEVICE_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <memory>
#include <vector>

namespace paddle {
namespace lite {

class metal_queue;
class metal_context;

class metal_device {
 public:
  std::shared_ptr<metal_queue> create_queue() const;
  std::shared_ptr<metal_queue> get_default_queue() const;

#if defined(__OBJC__)
  id<MTLDevice> get_device() const;
  void set_device(id<MTLDevice> device);
#else
  void *get_device() const;
  void set_device(void *device);
#endif
  void set_context(metal_context *context);
  void set_name(const char *name);

 private:
#if defined(__OBJC__)
  id<MTLDevice> device_;
#else
  void *device_;
#endif
  metal_context *context_;
  mutable std::vector<std::shared_ptr<metal_queue>> queues_;
  const char *name_;
};
}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_DEVICE_H_
