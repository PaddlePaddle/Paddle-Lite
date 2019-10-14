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
#include <map>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/macros.h"

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/target_wrapper.h"
#endif  // LITE_WITH_OPENCL

#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/target_wrapper.h"
#endif  // LITE_WITH_CUDA

namespace paddle {
namespace lite {

// Malloc memory for a specific Target. All the targets should be an element in
// the `switch` here.
LITE_API void* TargetMalloc(TargetType target, size_t size);

// Free memory for a specific Target. All the targets should be an element in
// the `switch` here.
void LITE_API TargetFree(TargetType target, void* data);

// Copy a buffer from host to another target.
void TargetCopy(TargetType target, void* dst, const void* src, size_t size);
#ifdef LITE_WITH_OPENCL
void TargetCopyImage2D(TargetType target,
                       void* dst,
                       const void* src,
                       const std::array<size_t, 2>& image_shape,
                       const std::array<size_t, 2>& image_pitch);
#endif  // LITE_WITH_OPENCL

template <TargetType Target>
void CopySync(void* dst, const void* src, size_t size, IoDirection dir) {
  switch (Target) {
    case TARGET(kX86):
    case TARGET(kHost):
    case TARGET(kARM):
      TargetWrapper<TARGET(kHost)>::MemcpySync(
          dst, src, size, IoDirection::HtoH);
      break;
#ifdef LITE_WITH_CUDA
    case TARGET(kCUDA):
      TargetWrapperCuda::MemcpySync(dst, src, size, dir);
      break;
#endif
#ifdef LITE_WITH_OPENCL
    case TargetType::kOpenCL:
      TargetWrapperCL::MemcpySync(dst, src, size, dir);
      break;
#endif  // LITE_WITH_OPENCL
#ifdef LITE_WITH_FPGA
    case TARGET(kFPGA):
      TargetWrapper<TARGET(kFPGA)>::MemcpySync(dst, src, size, dir);
      break;
#endif
  }
}

// Memory buffer manager.
class Buffer {
 public:
  Buffer() = default;
  Buffer(TargetType target, size_t size) : space_(size), target_(target) {}

  void* data() const { return data_; }
  TargetType target() const { return target_; }
  size_t space() const { return space_; }

  void ResetLazy(TargetType target, size_t size) {
    if (target != target_ || space_ < size) {
      Free();
      data_ = TargetMalloc(target, size);
      target_ = target;
      space_ = size;
    }
  }

  void ResizeLazy(size_t size) { ResetLazy(target_, size); }

#ifdef LITE_WITH_OPENCL
  template <typename T>
  void ResetLazyImage2D(TargetType target,
                        const std::array<size_t, 2>& image2d_shape) {
    size_t size = sizeof(T) * image2d_shape[0] * image2d_shape[1] *
                  4;  // 4 for RGBA, un-used for opencl Image2D
    VLOG(4) << "cl_image2d_shape_[width, height]:" << cl_image2d_shape_["width"]
            << " " << cl_image2d_shape_["height"];
    if (target != target_ || cl_image2d_shape_["width"] < image2d_shape[0] ||
        cl_image2d_shape_["height"] < image2d_shape[1]) {
      Free();
      data_ = TargetWrapperCL::MallocImage<T>(image2d_shape);
      target_ = target;
      space_ = size;  // un-used for opencl Image2D
      cl_image2d_shape_ = {{"width", image2d_shape[0]},
                           {"height", image2d_shape[1]}};
    }
  }

// TODO(ysh329): not used ResizeLazyImage2D for opencl Image2D
//  template <typename T>
//  void ResizeLazyImage2D(const std::map<std::string, size_t>& image2d_shape) {
//    ResetLazyImage2D<T>(target_, image2d_shape);
//  }

// TODO(ysh329): not used CopyImage2DFrom for opencl Image2D
//  template <typename T>
//  void CopyImage2DFrom(const Buffer& other,
//                       const std::map<std::string, size_t>& image2d_shape,
//                       const std::array<size_t, 2>& image2d_pitch) {
//    target_ = other.target_;
//    ResizeLazyImage2D<T>(image2d_shape, image2d_pitch);
//    TargetCopyImage2D(
//        target_, data_, other.data_, image2d_shape, image2d_pitch);
//  }
#endif

  void Free() {
    if (space_ > 0) {
      TargetFree(target_, data_);
    }
    target_ = TargetType::kHost;
    space_ = 0;
  }

  void CopyDataFrom(const Buffer& other, size_t nbytes) {
    target_ = other.target_;
    ResizeLazy(nbytes);
    // TODO(Superjomn) support copy between different targets.
    TargetCopy(target_, data_, other.data_, nbytes);
  }

  ~Buffer() { Free(); }

 private:
  // memory it actually malloced.
  size_t space_{0};
  std::map<std::string, size_t>
      cl_image2d_shape_{};  // used for opencl cl::Image2D
  void* data_{nullptr};
  TargetType target_{TargetType::kHost};
};

}  // namespace lite
}  // namespace paddle
