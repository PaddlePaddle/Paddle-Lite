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

#include "lite/backends/metal/metal_kernel.h"
#include "lite/backends/metal/metal_queue.h"
#include <algorithm>

namespace paddle {
namespace lite {

MetalKernel::MetalKernel(const MetalKernelProgram kernel) : program_(kernel) {}

void MetalKernel::Execute(const MetalEncoder& encoder,
                          const MetalUint3& global_work_size,
                          const int groupDepth) {
  auto slices = (global_work_size.z * 4 + 3) / 4;

  auto width = (MetalUint)program_.pipeline_state_.threadExecutionWidth;
  auto height = (MetalUint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
  auto threads_per_group = MTLSizeMake(width, height, 1);

  auto group_width = (global_work_size.x + width - 1) / width;
  auto group_height = (global_work_size.y + height - 1) / height;

  MTLSize groups = MTLSizeMake(group_width, group_height, groupDepth ? groupDepth : slices);
  assert(groups.width > 0 && groups.height > 0 && groups.depth > 0);

  [encoder.metal_command_encoder_ dispatchThreadgroups:groups
                                 threadsPerThreadgroup:threads_per_group];
  [encoder.metal_command_encoder_ endEncoding];
  return;
}

void MetalKernel::Execute(const MetalEncoder& encoder,
                          const MetalUint3& texture_array_3d,
                          bool quadruple) {
  auto slices = (texture_array_3d.z * 4 + 3) / 4;
  int width = 0, height = 0, group_width = 0, group_height = 0;

  if (quadruple) {
    width = (int)program_.pipeline_state_.threadExecutionWidth / 4;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.threadExecutionWidth / width;
    height = std::min<int>(height, texture_array_3d.y);
    group_width = (texture_array_3d.x / 2 + width - 1) / width;
    group_height = (texture_array_3d.y / 2 + height - 1) / height;
  } else {
    width = (int)program_.pipeline_state_.threadExecutionWidth;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
    height = std::min<int>(height, texture_array_3d.y);
    group_width = (texture_array_3d.x + width - 1) / width;
    group_height = (texture_array_3d.y + height - 1) / height;
  }

  auto threads_per_group = MTLSizeMake(width, height, 1);
  MTLSize groups = MTLSizeMake(group_width, group_height, slices);

  [encoder.metal_command_encoder_ dispatchThreadgroups:groups
                                 threadsPerThreadgroup:threads_per_group];
  [encoder.metal_command_encoder_ endEncoding];
  return;
}

MetalKernelProgram::~MetalKernelProgram(){
}
}
}
