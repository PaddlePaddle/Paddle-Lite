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

#ifndef LITE_BACKENDS_METAL_METAL_CONTEXT_H_
#define LITE_BACKENDS_METAL_METAL_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_device.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/backends/metal/metal_kernel.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {
class RuntimeProgram;

class MetalContext {
 public:
  /// device
  void PrepareDevices();
  int GetDevicesNum();
  MetalDevice* GetDeviceByID(int id);
  const MetalDevice* GetDefaultDevice();

  void CreateCommandBuffer(RuntimeProgram* program = nullptr);
  void WaitUntilCompleted();

  void set_metal_path(std::string path);
  void set_use_aggressive_optimization(bool flag);
  void set_use_mps(bool flag);
  bool use_mps() const { return use_mps_; }
  bool use_aggressive_optimization() const {
    return use_aggressive_optimization_;
  }

  /// queue
  std::shared_ptr<MetalQueue> GetDefaultQueue(const MetalDevice& device);
  std::shared_ptr<MetalQueue> CreateQueue(const MetalDevice& device);

  /// program
  std::shared_ptr<MetalKernel> GetKernel(const MetalDevice& device,
                                         const std::string function_name);

  void CreateLibraryWithFile(const MetalDevice& device,
                             std::string library_name = "");

  /// buffer_and_image
  std::shared_ptr<MetalBuffer> CreateBuffer(
      const MetalDevice& device,
      size_t length,
      METAL_ACCESS_FLAG flags = METAL_ACCESS_FLAG::CPUReadWrite);

  std::shared_ptr<MetalBuffer> CreateBuffer(
      const MetalDevice& device,
      void* data,
      size_t length,
      METAL_ACCESS_FLAG flags = METAL_ACCESS_FLAG::CPUReadWrite);

  MetalDevice* best_metal_device_{nullptr};
  mutable std::vector<std::shared_ptr<MetalDevice>> devices_ = {};

  std::unique_ptr<MetalCommandBuffer> cmd_buf_;

#if defined(__OBJC__)
  id<MTLLibrary> library_ = nil;
  std::map<size_t, id<MTLLibrary>> library_map_;
#else
  void* library_ = nullptr;
  std::map<size_t, void*> library_map_;
#endif

 private:
  bool got_devices_{false};
  std::string metal_path_;
  bool use_aggressive_optimization_{false};
  bool use_mps_{false};
  RuntimeProgram* program_ = nullptr;
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_CONTEXT_H_
