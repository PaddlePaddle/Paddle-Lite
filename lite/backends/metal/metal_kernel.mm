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

#include <algorithm>
#include <vector>

#include "lite/backends/metal/metal_arguments_helper.h"
#include "lite/backends/metal/metal_kernel.h"
#include "lite/backends/metal/metal_kernel_arg.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {

static std::unique_ptr<MetalKernel::MetalEncoder> CreateEncoder(
    const MetalQueue& queue, const MetalKernel::MetalKernelProgram& entry) {
  id<MTLCommandBuffer> cmd_buffer = queue.CreateCommandBuffer();
  auto encoder = new MetalKernel::MetalEncoder();
  encoder->metal_command_buffer_ = cmd_buffer;
  encoder->metal_command_encoder_ = [cmd_buffer computeCommandEncoder];
  std::unique_ptr<MetalKernel::MetalEncoder> ret(encoder);
  [ret->metal_command_encoder_ setComputePipelineState:entry.pipeline_state_];
  return ret;
}

MetalUint3 MetalKernel::FixThreadgroupSize(const MetalKernel::MetalKernelProgram& program,
                                           const MetalUint3& original_local_work_size) const {
  MetalUint3 new_local_work_size = original_local_work_size;
  new_local_work_size.MaxThan1();
  const auto workgroup_total_size =
      new_local_work_size.x * new_local_work_size.y * new_local_work_size.z;
  auto max_total_local_size = program.pipeline_state_.maxTotalThreadsPerThreadgroup;

  if (max_total_local_size > 0 && workgroup_total_size > max_total_local_size) {
    if (original_local_work_size.y > 1 && max_total_local_size > 1) {
      new_local_work_size = {(uint32_t)(max_total_local_size / 2u), 2, 1};
    } else {
      new_local_work_size = {(uint32_t)max_total_local_size, 1, 1};
    }
  }
  return new_local_work_size;
}

void MetalKernel::Execute(const MetalQueue& queue,
                          const MetalUint3 global_work_size,
                          const MetalUint3 local_work_size,
                          const std::vector<MetalKernelArgument>& args) {
  const auto thread_group_dim = FixThreadgroupSize(program_, local_work_size);
  auto encoder = CreateEncoder(queue, program_);
  metal_argument_helper::ParseArguments(encoder->metal_command_encoder_, args);

  int dim = 3;
  if (global_work_size.y == 1 && global_work_size.z == 1) {
    dim = 1;
  } else if (global_work_size.y == 1 || global_work_size.z == 1) {
    dim = 2;
  }

  const MetalUint3 grid_dim_overflow{
      dim >= 1 && global_work_size.x > 0
          ? std::min(uint32_t(global_work_size.x % thread_group_dim.x), 1u)
          : 0u,
      dim >= 2 && global_work_size.y > 0
          ? std::min(uint32_t(global_work_size.y % thread_group_dim.y), 1u)
          : 0u,
      dim >= 3 && global_work_size.z > 0
          ? std::min(uint32_t(global_work_size.z % thread_group_dim.z), 1u)
          : 0u};
  MetalUint3 grid_dim{(global_work_size.x / thread_group_dim.x) + grid_dim_overflow.x,
                      global_work_size.y / thread_group_dim.y + grid_dim_overflow.y,
                      global_work_size.z / thread_group_dim.z + grid_dim_overflow.z};

  grid_dim.MaxThan1();

  const MTLSize metal_grid_dim{grid_dim.x, grid_dim.y, grid_dim.z};
  const MTLSize metal_block_dim{thread_group_dim.x, thread_group_dim.y, thread_group_dim.z};

  [encoder->metal_command_encoder_ dispatchThreadgroups:metal_grid_dim
                                  threadsPerThreadgroup:metal_block_dim];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

MetalKernel::MetalKernel(const MetalKernel::MetalKernelProgram kernel) : program_(kernel) {}

MetalUint3 MetalKernel::CaculateThreadsPerGroup(MetalUint3 t,
                                                MetalUint thread_execution_width,
                                                bool keep_z) {
  auto pwarp = SmallestLog2(thread_execution_width);
  auto px = SmallestLog2(t.x);
  auto sx = ceil(log2(t.x));
  auto py = SmallestLog2(t.y);
  auto sy = ceil(log2(t.y));

  // accurately match on x
  if (px >= pwarp) {
    return {thread_execution_width, 1, 1};
  }
  // accurately match on xy
  else if (px + py >= pwarp && sx < pwarp / 2) {
    MetalUint x = pow(2, px);
    return {x, thread_execution_width / x, 1};
  }
  // similarly match on x
  else if (sx >= pwarp) {
    return {thread_execution_width, 1, 1};
  }
  // similarly match on xy
  else if (sx + sy >= pwarp) {
    MetalUint x = pow(2, sx);
    return {x, thread_execution_width / x, 1};
  }

  // on xyz (for most shaders do not protect gid.z, z axis must be accurately match)
  auto pz = SmallestLog2(t.z);
  auto sz = keep_z ? ceil(log2(t.z)) : pz;
  if (px + py + pz >= pwarp) {
    MetalUint x = pow(2, px), y = pow(2, py);
    return {x, y, thread_execution_width / x / y};
  } else if (sx + sy + sz >= pwarp) {
    MetalUint x = pow(2, sx), z = pow(2, MIN(sz, pwarp - sx));
    return {x, thread_execution_width / x / z, z};
  } else {
    MetalUint z = pow(2, sz);
    return {t.x, t.y, z};
  }
}

void MetalKernel::Execute(const MetalQueue& queue,
                          const MetalUint3& global_work_size,
                          const int groupDepth,
                          std::vector<std::pair<MetalKernelArgument, int>> args) {
  auto slices = (global_work_size.z * 4 + 3) / 4;

  auto encoder = CreateEncoder(queue, program_);
  auto ret = metal_argument_helper::ParseArguments(encoder->metal_command_encoder_, args);
  if (ret == false) exit(-3);

  auto width = (MetalUint)program_.pipeline_state_.threadExecutionWidth;
  auto height = (MetalUint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
  auto threads_per_group = MTLSizeMake(width, height, 1);

  auto group_width = (global_work_size.x + width - 1) / width;
  auto group_height = (global_work_size.y + height - 1) / height;

  MTLSize groups = MTLSizeMake(group_width, group_height, groupDepth ? groupDepth : slices);
  assert(groups.width > 0 && groups.height > 0 && groups.depth > 0);

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

void MetalKernel::Execute(const MetalQueue& queue,
                          const MetalUint3& texture_array_3d,
                          bool quadruple,
                          std::vector<std::pair<MetalKernelArgument, int>> args) {
  auto slices = (texture_array_3d.z * 4 + 3) / 4;
  int width = 0, height = 0, group_width = 0, group_height = 0;
  auto encoder = CreateEncoder(queue, program_);
  auto ret = metal_argument_helper::ParseArguments(encoder->metal_command_encoder_, args);
  assert(ret == true);
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

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

void MetalKernel::Execute(const MetalQueue& queue,
                          const MetalUint3& texture_array_3d,
                          bool quadruple,
                          std::vector<MetalKernelArgument> args) {
  auto slices = (texture_array_3d.z * 4 + 3) / 4;
  int width = 0, height = 0, group_width = 0, group_height = 0;
  auto encoder = CreateEncoder(queue, program_);
  auto ret = metal_argument_helper::ParseArguments(encoder->metal_command_encoder_, args);
  assert(ret == true);
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

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

}
}
