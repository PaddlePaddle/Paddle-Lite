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

#include <Metal/Metal.h>
#include "lite/utils/cp_logging.h"
#include "lite/backends/metal/metal_context.h"
#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_device.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/backends/metal/metal_kernel.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {

void metal_context::prepare_devices() {
  if (got_devices_) return;
  devices_.clear();
#if defined(TARGET_IOS)
  id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
  if (!mtl_device) return;
  NSArray<id<MTLDevice>>* mtl_devices =
      (NSArray<id<MTLDevice>>*)[NSArray arrayWithObjects:mtl_device, nil];
#else
  NSArray<id<MTLDevice>>* mtl_devices = MTLCopyAllDevices();
#endif
  if (!mtl_devices) return;

  uint32_t device_num = 0;
  for (id<MTLDevice> dev in mtl_devices) {
    devices_.emplace_back(std::shared_ptr<metal_device>(new metal_device));
    auto& device = (metal_device&)*devices_.back();
    device.set_device(dev);
    device.set_context(this);
    device.set_name([[dev name] UTF8String]);
    ++device_num;
  }

  best_metal_device_ = devices_[0].get();
}

int metal_context::get_devices_num() {
  if (!got_devices_) {
    prepare_devices();
    got_devices_ = true;
  }
  return devices_.size();
}

metal_device* metal_context::get_device_by_id(int id) {
  if (!got_devices_) {
    prepare_devices();
    got_devices_ = true;
  }
  if (id < devices_.size())
    return devices_[id].get();
  else {
    LOG(ERROR) << "ERROR: cannot find the metal device id " << id << "in our device"
               << "\n";
    return nullptr;
  }
}

const metal_device* metal_context::get_default_device() {
  if (!got_devices_) {
    prepare_devices();
    got_devices_ = true;
  }
  return best_metal_device_;
}

void metal_context::set_metal_path(std::string path) { metal_path_ = path; }

void metal_context::set_use_aggressive_optimization(bool flag) {
  use_aggressive_optimization_ = flag;
}

void metal_context::set_use_mps(bool flag) { use_mps_ = flag; }

__unused std::shared_ptr<metal_queue> metal_context::create_queue(const metal_device& device) {
  return device.create_queue();
}

std::shared_ptr<metal_queue> metal_context::get_default_queue(const metal_device& device) {
  return device.get_default_queue();
}

std::shared_ptr<metal_kernel> metal_context::get_kernel(const metal_device& device,
                                                        const std::string function_name,
                                                        const std::string library_path) {
  std::string library_name = library_path;
  if (library_name.empty()) {
    library_name = metal_path_;
  }

  NSError* error = Nil;
  id<MTLLibrary> library =
      [device.get_device() newLibraryWithFile:[NSString stringWithUTF8String:library_name.c_str()]
                                        error:&error];
  if (!library) {
    auto err_str = error != nullptr ? [[error localizedDescription] UTF8String] : "unknown error";
    LOG(ERROR) << "ERROR: load library error: " << err_str << "\n";
    return std::shared_ptr<metal_kernel>{};
  }

  metal_kernel::metal_kernel_program program;
  const auto func_name = [NSString stringWithUTF8String:function_name.c_str()];
  program.function_ = [library newFunctionWithName:func_name];
  if (!program.function_) {
    auto err_str = error != nullptr ? [[error localizedDescription] UTF8String] : "unknown error";
    LOG(ERROR) << "ERROR: load function " << function_name << "from library error: " << err_str
               << "\n";
    return std::shared_ptr<metal_kernel>{};
  }

  program.pipeline_state_ =
      [device.get_device() newComputePipelineStateWithFunction:program.function_ error:&error];
  if (!program.pipeline_state_) {
    auto err_str = error != nullptr ? [[error localizedDescription] UTF8String] : "unknown error";
    LOG(ERROR) << "ERROR: failed to create pipeline state" << function_name << err_str << "\n";
    return std::shared_ptr<metal_kernel>{};
  }

  auto ret = std::make_shared<metal_kernel>(program);
  return ret;
}

std::shared_ptr<metal_buffer> metal_context::create_buffer(const metal_device& device,
                                                           size_t length,
                                                           const METAL_ACCESS_FLAG flags) {
  return std::make_shared<metal_buffer>(device, length, flags);
}

std::shared_ptr<metal_buffer> metal_context::create_buffer(const metal_device& device,
                                                           void* data,
                                                           size_t length,
                                                           const METAL_ACCESS_FLAG flags) {
  return std::make_shared<metal_buffer>(device, data, length, flags);
}

}
}