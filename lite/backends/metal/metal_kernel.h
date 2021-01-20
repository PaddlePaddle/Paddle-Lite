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

#ifndef LITE_BACKENDS_METAL_METAL_KERNEL_H_
#define LITE_BACKENDS_METAL_METAL_KERNEL_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <utility>
#include <vector>

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_kernel_arg.h"

namespace paddle {
namespace lite {

class metal_queue;
class metal_kernel {
 public:
#if defined(__OBJC__)
  struct metal_encoder {
    id<MTLCommandBuffer> metal_command_buffer_{nil};
    id<MTLComputeCommandEncoder> metal_command_encoder_{nil};
  };
  struct metal_kernel_program {
    id<MTLFunction> function_{nil};
    id<MTLComputePipelineState> pipeline_state_{nil};
  };
#else
  struct metal_encoder {
    void* metal_command_buffer_{nullptr};
    void* metal_command_encoder_{nullptr};
  };

  struct metal_kernel_program {
    void* function_{nullptr};
    void* pipeline_state_{nullptr};
  };
#endif
  metal_kernel_program program_;
  explicit metal_kernel(const metal_kernel_program kernel);
  ~metal_kernel() = default;

 public:
  __unused void execute_use_zkeeped(
      const metal_queue& queue,
      const metal_uint3 global_work_size,
      bool z_keep,
      std::vector<std::pair<metal_kernel_arg, int>> args);

  __unused void execute_thread_group(
      const metal_queue& queue,
      const metal_uint3& global_work_size,
      const metal_uint3& group_size,
      std::vector<std::pair<metal_kernel_arg, int>> args);

  void execute(const metal_queue& queue,
               const metal_uint3& texture_array_3d,
               const int groupDepth,
               std::vector<std::pair<metal_kernel_arg, int>> args);

  void execute(const metal_queue& queue,
               const metal_uint3& texture_array_3d,
               bool quadruple,
               std::vector<std::pair<metal_kernel_arg, int>> args);

  void execute(const metal_queue& queue,
               const metal_uint3& texture_array_3d,
               bool quadruple,
               std::vector<metal_kernel_arg> args);

 private:
  template <typename... Args>
  void execute(const metal_queue& queue,
               const metal_uint3 global_work_size,
               const metal_uint3 local_work_size,
               Args... args) {
    execute(queue, global_work_size, local_work_size, {args...});
  }

  //  template <typename... Args>
  //  void execute(const metal_queue& queue,
  //               const metal_uint3 global_work_size,
  //               Args... args) {
  //    execute(queue, global_work_size, {args...});
  //  }

  template <typename... Args>
  void execute2(const metal_queue& queue,
                const metal_uint3 global_work_size,
                const metal_uint3 local_work_size,
                Args... args) {
    execute2(queue, global_work_size, local_work_size, {args...});
  }

  template <typename... Args>
  void execute2(const metal_queue& queue,
                const metal_uint3 global_work_size,
                bool z_keep,
                Args... args) {
    execute2(queue, global_work_size, z_keep, {args...});
  }

  template <typename... Args>
  void execute3(const metal_queue& queue,
                const metal_uint3 global_work_size,
                std::vector<int> offsets,
                bool z_keep,
                Args... args) {
    execute3(queue, global_work_size, offsets, z_keep, {args...});
  }

  void execute3(const metal_queue& queue,
                const metal_uint3 global_work_size,
                std::vector<int> offsets,
                bool z_keep,
                std::vector<metal_kernel_arg> args);

  void execute(const metal_queue& queue,
               const metal_uint3 global_work_size,
               const metal_uint3 local_work_size,
               const std::vector<metal_kernel_arg>& args);

  void execute2(const metal_queue& queue,
                const metal_uint3 global_work_size,
                const metal_uint3 local_work_size,
                const std::vector<metal_kernel_arg>& args);

  void execute2(const metal_queue& queue,
                const metal_uint3 global_work_size,
                bool z_keep,
                const std::vector<metal_kernel_arg>& args);

  metal_uint3 fix_threadgroup_size(
      const metal_kernel::metal_kernel_program& program,
      const metal_uint3& original_local_work_size) const;

  metal_uint3 caculate_threads_per_group(metal_uint3 t,
                                         metal_uint threadExecutionWidth,
                                         bool keep_z);
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_KERNEL_H_
