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

#include <memory>
#include <string>
#include <vector>

#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_device.h"
#include "lite/backends/metal/metal_kernel.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {

class metal_context {
 public:
  /// device
  void prepare_devices();
  int get_devices_num();
  metal_device* get_device_by_id(int id);
  const metal_device* get_default_device();

  void set_metal_path(std::string path);
  void set_use_aggressive_optimization(bool flag);
  void set_use_mps(bool flag);

  bool get_use_mps() const { return use_mps_; }
  bool get_use_aggressive_optimization() const {
    return use_aggressive_optimization_;
  }

  /// queue
  std::shared_ptr<metal_queue> get_default_queue(const metal_device& device);
  std::shared_ptr<metal_queue> create_queue(const metal_device& device);

  /// program
  std::shared_ptr<metal_kernel> get_kernel(const metal_device& device,
                                           std::string function_name,
                                           std::string library_name = "");

  /// buffer_and_image
  std::shared_ptr<metal_buffer> create_buffer(
      const metal_device& device,
      size_t length,
      METAL_ACCESS_FLAG flags = METAL_ACCESS_FLAG::CPUReadWrite);

  std::shared_ptr<metal_buffer> create_buffer(
      const metal_device& device,
      void* data,
      size_t length,
      METAL_ACCESS_FLAG flags = METAL_ACCESS_FLAG::CPUReadWrite);

  metal_device* best_metal_device_{nullptr};
  mutable std::vector<std::shared_ptr<metal_device>> devices_ = {};

 private:
  bool got_devices_{false};
  std::string metal_path_;
  bool use_aggressive_optimization_{false};
  bool use_mps_{false};
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_CONTEXT_H_
