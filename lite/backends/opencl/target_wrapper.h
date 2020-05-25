// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "lite/backends/opencl/cl_include.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperCL =
    TargetWrapper<TARGET(kOpenCL), cl::CommandQueue, cl::Event>;
// This interface should be specified by each kind of target.
template <>
class TargetWrapper<TARGET(kOpenCL), cl::CommandQueue, cl::Event> {
 public:
  using stream_t = cl::CommandQueue;
  using event_t = cl::Event;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  template <typename R>
  static void* MallocImage(const size_t cl_image2d_width,
                           const size_t cl_image2d_height,
                           void* host_ptr = nullptr);
  static void FreeImage(void* image);

  static void* Map(void* buffer, size_t offset, size_t size);
  static void* MapImage(void* image,
                        const size_t cl_image2d_width,
                        const size_t cl_image2d_height,
                        const size_t cl_image2d_row_pitch,
                        const size_t cl_image2d_slice_pitch);
  static void Unmap(void* cl_obj, void* mapped_ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream);
  static void ImgcpySync(void* dst,
                         const void* src,
                         const size_t cl_image2d_width,
                         const size_t cl_image2d_height,
                         const size_t cl_image2d_row_pitch,
                         const size_t cl_image2d_slice_pitch,
                         IoDirection dir);
  static void ImgcpyAsync(void* dst,
                          const void* src,
                          const size_t cl_image2d_width,
                          const size_t cl_image2d_height,
                          const size_t cl_image2d_row_pitch,
                          const size_t cl_image2d_slice_pitch,
                          IoDirection dir,
                          const stream_t& stream);
};

}  // namespace lite
}  // namespace paddle
