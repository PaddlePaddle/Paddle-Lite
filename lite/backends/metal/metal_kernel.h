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

#ifndef LITE_BACKENDS_METAL_METAL_KERNEL_H_
#define LITE_BACKENDS_METAL_METAL_KERNEL_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <utility>
#include <vector>

#include "lite/backends/metal/metal_common.h"

namespace paddle {
namespace lite {

class MetalQueue;
class MetalEncoder;

struct MetalKernelProgram {
#if defined(__OBJC__)
  id<MTLFunction> function_{nil};
  id<MTLComputePipelineState> pipeline_state_{nil};
#else
  void* function_{nullptr};
  void* pipeline_state_{nullptr};
#endif

  virtual ~MetalKernelProgram();
};

class MetalKernel {
 public:
  MetalKernelProgram program_;
  explicit MetalKernel(const MetalKernelProgram kernel);
  ~MetalKernel() = default;

 public:
  void Execute(const MetalEncoder& encoder,
               const MetalUint3& texture_array_3d,
               const int groupDepth);

  void Execute(const MetalEncoder& encoder,
               const MetalUint3& texture_array_3d,
               bool quadruple);
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_KERNEL_H_
