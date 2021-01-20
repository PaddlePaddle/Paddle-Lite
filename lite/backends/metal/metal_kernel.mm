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
using namespace std;

static unique_ptr<metal_kernel::metal_encoder> create_encoder(
    const metal_queue& queue, const metal_kernel::metal_kernel_program& entry) {
  id<MTLCommandBuffer> cmd_buffer = queue.create_command_buffer();
  auto encoder = new metal_kernel::metal_encoder();
  encoder->metal_command_buffer_ = cmd_buffer;
  encoder->metal_command_encoder_ = [cmd_buffer computeCommandEncoder];
  unique_ptr<metal_kernel::metal_encoder> ret(encoder);
  [ret->metal_command_encoder_ setComputePipelineState:entry.pipeline_state_];
  return ret;
}

metal_uint3 metal_kernel::fix_threadgroup_size(const metal_kernel::metal_kernel_program& program,
                                               const metal_uint3& original_local_work_size) const {
  metal_uint3 new_local_work_size = original_local_work_size;
  new_local_work_size.max_than_1();
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

void metal_kernel::execute(const metal_queue& queue,
                           const metal_uint3 global_work_size,
                           const metal_uint3 local_work_size,
                           const std::vector<metal_kernel_arg>& args) {
  const auto thread_group_dim = fix_threadgroup_size(program_, local_work_size);
  auto encoder = create_encoder(queue, program_);
  metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);

  int dim = 3;
  if (global_work_size.y == 1 && global_work_size.z == 1) {
    dim = 1;
  } else if (global_work_size.y == 1 || global_work_size.z == 1) {
    dim = 2;
  }

  const metal_uint3 grid_dim_overflow{
      dim >= 1 && global_work_size.x > 0
          ? std::min(uint32_t(global_work_size.x % thread_group_dim.x), 1u)
          : 0u,
      dim >= 2 && global_work_size.y > 0
          ? std::min(uint32_t(global_work_size.y % thread_group_dim.y), 1u)
          : 0u,
      dim >= 3 && global_work_size.z > 0
          ? std::min(uint32_t(global_work_size.z % thread_group_dim.z), 1u)
          : 0u};
  metal_uint3 grid_dim{(global_work_size.x / thread_group_dim.x) + grid_dim_overflow.x,
                       global_work_size.y / thread_group_dim.y + grid_dim_overflow.y,
                       global_work_size.z / thread_group_dim.z + grid_dim_overflow.z};

  grid_dim.max_than_1();

  const MTLSize metal_grid_dim{grid_dim.x, grid_dim.y, grid_dim.z};
  const MTLSize metal_block_dim{thread_group_dim.x, thread_group_dim.y, thread_group_dim.z};

  [encoder->metal_command_encoder_ dispatchThreadgroups:metal_grid_dim
                                  threadsPerThreadgroup:metal_block_dim];
  //    MTLSize global_id = {global_work_size.x, global_work_size.y, global_work_size.z};
  //    [encoder->metal_command_encoder_ dispatchThreads:global_id
  //    threadsPerThreadgroup:metal_block_dim];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

void metal_kernel::execute2(const metal_queue& queue,
                            const metal_uint3 global_work_size,
                            const metal_uint3 local_work_size,
                            const vector<metal_kernel_arg>& args) {
  auto encoder = create_encoder(queue, program_);
  metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  uint32_t max_total_threads_per_threadgroup =
      (uint32_t)program_.pipeline_state_.maxTotalThreadsPerThreadgroup;

  MTLSize threads;
  MTLSize threads_per_group;
  threads.width = global_work_size.x;
  threads.height = global_work_size.y;
  threads.depth = global_work_size.z;

  threads_per_group.width =
      std::max<uint32_t>(local_work_size.x, max_total_threads_per_threadgroup);
  threads_per_group.height =
      std::max<uint32_t>(local_work_size.y, max_total_threads_per_threadgroup);
  threads_per_group.depth =
      std::max<uint32_t>(local_work_size.z, max_total_threads_per_threadgroup);

  //#if defined(TARGET_IOS)
  //  if (@available(iOS 11.0, *)) {
  //    if ([device_ supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
  //      [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
  //      return;
  //    }
  //  }
  //#endif
  MTLSize groups = {
      UP_DIV(threads.width, threads_per_group.width),
      UP_DIV(threads.height, threads_per_group.height),
      UP_DIV(threads.depth, threads_per_group.depth),
  };

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

void metal_kernel::execute2(const metal_queue& queue,
                            const metal_uint3 global_work_size,
                            bool z_keep,
                            const vector<metal_kernel_arg>& args) {
  auto encoder = create_encoder(queue, program_);
  metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  metal_uint max_total_threads_per_threadgroup =
      (metal_uint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup;
  metal_uint thread_execution_width = (metal_uint)program_.pipeline_state_.threadExecutionWidth;
  auto local_work_size =
      caculate_threads_per_group(global_work_size, thread_execution_width, z_keep);
  MTLSize threads;
  MTLSize threads_per_group;
  threads.width = global_work_size.x;
  threads.height = global_work_size.y;
  threads.depth = global_work_size.z;

  threads_per_group.width =
      std::min<uint32_t>(local_work_size.x, max_total_threads_per_threadgroup);
  threads_per_group.height =
      std::min<uint32_t>(local_work_size.y, max_total_threads_per_threadgroup);
  threads_per_group.depth =
      std::min<uint32_t>(local_work_size.z, max_total_threads_per_threadgroup);

  //#if defined(TARGET_IOS)
  //  if (@available(iOS 11.0, *)) {
  //    if ([_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
  //      [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
  //      return;
  //    }
  //  }
  //#endif
  MTLSize groups = {
      UP_DIV(threads.width, threads_per_group.width),
      UP_DIV(threads.height, threads_per_group.height),
      UP_DIV(threads.depth, threads_per_group.depth),
  };

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

metal_kernel::metal_kernel(const metal_kernel::metal_kernel_program kernel) : program_(kernel) {}

metal_uint3 metal_kernel::caculate_threads_per_group(metal_uint3 t,
                                                     metal_uint threadExecutionWidth,
                                                     bool keep_z) {
  auto pwarp = smallest_log2(threadExecutionWidth);
  auto px = smallest_log2(t.x);
  auto sx = ceil(log2(t.x));
  auto py = smallest_log2(t.y);
  auto sy = ceil(log2(t.y));

  // accurately match on x
  if (px >= pwarp) {
    return {threadExecutionWidth, 1, 1};
  }
  // accurately match on xy
  else if (px + py >= pwarp && sx < pwarp / 2) {
    metal_uint x = pow(2, px);
    return {x, threadExecutionWidth / x, 1};
  }
  // similarly match on x
  else if (sx >= pwarp) {
    return {threadExecutionWidth, 1, 1};
  }
  // similarly match on xy
  else if (sx + sy >= pwarp) {
    metal_uint x = pow(2, sx);
    return {x, threadExecutionWidth / x, 1};
  }

  // on xyz (for most shaders do not protect gid.z, z axis must be accurately match)
  auto pz = smallest_log2(t.z);
  auto sz = keep_z ? ceil(log2(t.z)) : pz;
  if (px + py + pz >= pwarp) {
    metal_uint x = pow(2, px), y = pow(2, py);
    return {x, y, threadExecutionWidth / x / y};
  } else if (sx + sy + sz >= pwarp) {
    metal_uint x = pow(2, sx), z = pow(2, MIN(sz, pwarp - sx));
    return {x, threadExecutionWidth / x / z, z};
  } else {
    metal_uint z = pow(2, sz);
    return {t.x, t.y, z};
  }
}

void metal_kernel::execute3(const metal_queue& queue,
                            const metal_uint3 global_work_size,
                            vector<int> offsets,
                            bool z_keep,
                            vector<metal_kernel_arg> args) {
  auto encoder = create_encoder(queue, program_);
  metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, offsets, args);
  metal_uint max_total_threads_per_threadgroup =
      (metal_uint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup;
  metal_uint thread_execution_width = (metal_uint)program_.pipeline_state_.threadExecutionWidth;
  auto local_work_size =
      caculate_threads_per_group(global_work_size, thread_execution_width, z_keep);
  MTLSize threads;
  MTLSize threads_per_group;
  threads.width = global_work_size.x;
  threads.height = global_work_size.y;
  threads.depth = global_work_size.z;

  threads_per_group.width =
      std::min<uint32_t>(local_work_size.x, max_total_threads_per_threadgroup);
  threads_per_group.height =
      std::min<uint32_t>(local_work_size.y, max_total_threads_per_threadgroup);
  threads_per_group.depth =
      std::min<uint32_t>(local_work_size.z, max_total_threads_per_threadgroup);

  //#if defined(TARGET_IOS)
  //  if (@available(iOS 11.0, *)) {
  //    if ([_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
  //      [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
  //      return;
  //    }
  //  }
  //#endif
  MTLSize groups = {
      UP_DIV(threads.width, threads_per_group.width),
      UP_DIV(threads.height, threads_per_group.height),
      UP_DIV(threads.depth, threads_per_group.depth),
  };

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

void metal_kernel::execute_use_zkeeped(const metal_queue& queue,
                                       const metal_uint3 global_work_size,
                                       bool z_keep,
                                       vector<pair<metal_kernel_arg, int>> args) {
  auto encoder = create_encoder(queue, program_);
  auto ret = metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  if (ret == false) exit(-3);
  metal_uint max_total_threads_per_threadgroup =
      (metal_uint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup;
  metal_uint thread_execution_width = (metal_uint)program_.pipeline_state_.threadExecutionWidth;
  auto local_work_size =
      caculate_threads_per_group(global_work_size, thread_execution_width, z_keep);
  MTLSize threads;
  MTLSize threads_per_group;
  threads.width = global_work_size.x;
  threads.height = global_work_size.y;
  threads.depth = global_work_size.z;

  threads_per_group.width =
      std::min<uint32_t>(local_work_size.x, max_total_threads_per_threadgroup);
  threads_per_group.height =
      std::min<uint32_t>(local_work_size.y, max_total_threads_per_threadgroup);
  threads_per_group.depth =
      std::min<uint32_t>(local_work_size.z, max_total_threads_per_threadgroup);

  //#if defined(TARGET_IOS)
  //  if (@available(iOS 11.0, *)) {
  //    if ([_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
  //      [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
  //      return;
  //    }
  //  }
  //#endif
  MTLSize groups = {
      UP_DIV(threads.width, threads_per_group.width),
      UP_DIV(threads.height, threads_per_group.height),
      UP_DIV(threads.depth, threads_per_group.depth),
  };

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threads_per_group];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

__unused void metal_kernel::execute_thread_group(const metal_queue& queue,
                                                 const metal_uint3& group_size,
                                                 const metal_uint3& threads_per_group,
                                                 vector<pair<metal_kernel_arg, int>> args) {
  auto encoder = create_encoder(queue, program_);
  auto ret = metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  if (ret == false) exit(-3);

  MTLSize groups = MTLSizeMake(group_size.x, group_size.y, group_size.z);
  assert(groups.width > 0 && groups.height > 0 && groups.depth > 0);
  MTLSize threadsPerGroup =
      MTLSizeMake(threads_per_group.x, threads_per_group.y, threads_per_group.z);
  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threadsPerGroup];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
}

void metal_kernel::execute(const metal_queue& queue,
                           const metal_uint3& global_work_size,
                           const int groupDepth,
                           vector<pair<metal_kernel_arg, int>> args) {
  auto slices = (global_work_size.z * 4 + 3) / 4;

  auto encoder = create_encoder(queue, program_);
  auto ret = metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  if (ret == false) exit(-3);

  auto width = (metal_uint)program_.pipeline_state_.threadExecutionWidth;
  auto height = (metal_uint)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
  auto threadsPerGroup = MTLSizeMake(width, height, 1);

  auto groupWidth = (global_work_size.x + width - 1) / width;
  auto groupHeight = (global_work_size.y + height - 1) / height;

  MTLSize groups = MTLSizeMake(groupWidth, groupHeight, groupDepth ? groupDepth : slices);
  assert(groups.width > 0 && groups.height > 0 && groups.depth > 0);

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threadsPerGroup];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

void metal_kernel::execute(const metal_queue& queue,
                           const metal_uint3& texture_array_3d,
                           bool quadruple,
                           vector<pair<metal_kernel_arg, int>> args) {
  auto slices = (texture_array_3d.z * 4 + 3) / 4;
  int width = 0, height = 0, groupWidth = 0, groupHeight = 0;
  auto encoder = create_encoder(queue, program_);
  auto ret = metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  assert(ret == true);
  if (quadruple) {
    width = (int)program_.pipeline_state_.threadExecutionWidth / 4;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.threadExecutionWidth / width;
    height = std::min<int>(height, texture_array_3d.y);
    groupWidth = (texture_array_3d.x / 2 + width - 1) / width;
    groupHeight = (texture_array_3d.y / 2 + height - 1) / height;
  } else {
    width = (int)program_.pipeline_state_.threadExecutionWidth;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
    height = std::min<int>(height, texture_array_3d.y);
    groupWidth = (texture_array_3d.x + width - 1) / width;
    groupHeight = (texture_array_3d.y + height - 1) / height;
  }

  auto threadsPerGroup = MTLSizeMake(width, height, 1);
  MTLSize groups = MTLSizeMake(groupWidth, groupHeight, slices);

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threadsPerGroup];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

void metal_kernel::execute(const metal_queue& queue,
                           const metal_uint3& texture_array_3d,
                           bool quadruple,
                           vector<metal_kernel_arg> args) {
  auto slices = (texture_array_3d.z * 4 + 3) / 4;
  int width = 0, height = 0, groupWidth = 0, groupHeight = 0;
  auto encoder = create_encoder(queue, program_);
  auto ret = metal_argument_helper::parse_arguments(encoder->metal_command_encoder_, args);
  assert(ret == true);
  if (quadruple) {
    width = (int)program_.pipeline_state_.threadExecutionWidth / 4;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.threadExecutionWidth / width;
    height = std::min<int>(height, texture_array_3d.y);
    groupWidth = (texture_array_3d.x / 2 + width - 1) / width;
    groupHeight = (texture_array_3d.y / 2 + height - 1) / height;
  } else {
    width = (int)program_.pipeline_state_.threadExecutionWidth;
    width = std::min<int>(width, texture_array_3d.x);
    height = (int)program_.pipeline_state_.maxTotalThreadsPerThreadgroup / width;
    height = std::min<int>(height, texture_array_3d.y);
    groupWidth = (texture_array_3d.x + width - 1) / width;
    groupHeight = (texture_array_3d.y + height - 1) / height;
  }

  auto threadsPerGroup = MTLSizeMake(width, height, 1);
  MTLSize groups = MTLSizeMake(groupWidth, groupHeight, slices);

  [encoder->metal_command_encoder_ dispatchThreadgroups:groups
                                  threadsPerThreadgroup:threadsPerGroup];
  [encoder->metal_command_encoder_ endEncoding];
  [encoder->metal_command_buffer_ commit];
  return;
}

}
}
