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
#include <algorithm>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/logging.h"
#include "lite/utils/macros.h"

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/target_wrapper.h"
#endif  // LITE_WITH_OPENCL

#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/target_wrapper.h"
#endif  // LITE_WITH_CUDA

#ifdef LITE_WITH_BM
#include "lite/backends/bm/target_wrapper.h"
#endif  // LITE_WITH_BM

#ifdef LITE_WITH_MLU
#include "lite/backends/mlu/target_wrapper.h"
#endif  // LITE_WITH_MLU

#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/target_wrapper.h"
#endif  // LITE_WITH_XPU

namespace paddle {
namespace lite {

// Malloc memory for a specific Target. All the targets should be an element in
// the `switch` here.
LITE_API void* TargetMalloc(TargetType target, size_t size);

// Free memory for a specific Target. All the targets should be an element in
// the `switch` here.
void LITE_API TargetFree(TargetType target,
                         void* data,
                         std::string free_flag = "");

// Copy a buffer from host to another target.
void TargetCopy(TargetType target, void* dst, const void* src, size_t size);
#ifdef LITE_WITH_OPENCL
void TargetCopyImage2D(TargetType target,
                       void* dst,
                       const void* src,
                       const size_t cl_image2d_width,
                       const size_t cl_image2d_height,
                       const size_t cl_image2d_row_pitch,
                       const size_t cl_image2d_slice_pitch);
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
#ifdef LITE_WITH_MLU
    case TARGET(kMLU):
      TargetWrapperMlu::MemcpySync(dst, src, size, dir);
      break;
#endif
#ifdef LITE_WITH_FPGA
    case TARGET(kFPGA):
      TargetWrapper<TARGET(kFPGA)>::MemcpySync(dst, src, size, dir);
      break;
#endif
#ifdef LITE_WITH_BM
    case TARGET(kBM):
      TargetWrapper<TARGET(kBM)>::MemcpySync(dst, src, size, dir);
      break;
#endif
#ifdef LITE_WITH_XPU
    case TARGET(kXPU):
      TargetWrapperXPU::MemcpySync(dst, src, size, dir);
      break;
#endif
    default:
      LOG(FATAL)
          << "The copy function of this target has not been implemented yet.";
  }
}

// Memory buffer manager.
class Buffer {
 public:
  Buffer() = default;
  Buffer(TargetType target, size_t size) : space_(size), target_(target) {}
  Buffer(void* data, TargetType target, size_t size)
      : space_(size), data_(data), own_data_(false), target_(target) {}

  void* data() const { return data_; }
  TargetType target() const { return target_; }
  size_t space() const { return space_; }
  bool own_data() const { return own_data_; }

  void ResetLazy(TargetType target, size_t size) {
    if (target != target_ || space_ < size) {
      CHECK_EQ(own_data_, true) << "Can not reset unowned buffer.";
      Free();
      data_ = TargetMalloc(target, size);
      target_ = target;
      space_ = size;
#ifdef LITE_WITH_OPENCL
      cl_use_image2d_ = false;
#endif
    }
  }

  void ResizeLazy(size_t size) { ResetLazy(target_, size); }

#ifdef LITE_WITH_OPENCL
  template <typename T>
  void ResetLazyImage2D(TargetType target,
                        const size_t img_w_req,
                        const size_t img_h_req,
                        void* host_ptr = nullptr) {
    if (target != target_ || cl_image2d_width_ < img_w_req ||
        cl_image2d_height_ < img_h_req || host_ptr != nullptr) {
      CHECK_EQ(own_data_, true) << "Can not reset unowned buffer.";
      cl_image2d_width_ = std::max(cl_image2d_width_, img_w_req);
      cl_image2d_height_ = std::max(cl_image2d_height_, img_h_req);
      Free();
      data_ = TargetWrapperCL::MallocImage<T>(
          cl_image2d_width_, cl_image2d_height_, host_ptr);
      target_ = target;
      space_ = sizeof(T) * cl_image2d_width_ * cl_image2d_height_ *
               4;  // un-used for opencl Image2D, 4 for RGBA,
      cl_use_image2d_ = true;
    }
  }
#endif

  void Free() {
    if (space_ > 0 && own_data_) {
      if (!cl_use_image2d_) {
        TargetFree(target_, data_);
      } else {
        TargetFree(target_, data_, "cl_use_image2d_");
      }
    }
    data_ = nullptr;
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
  bool cl_use_image2d_{false};   // only used for OpenCL Image2D
  size_t cl_image2d_width_{0};   // only used for OpenCL Image2D
  size_t cl_image2d_height_{0};  // only used for OpenCL Image2D
  void* data_{nullptr};
  bool own_data_{true};
  TargetType target_{TargetType::kHost};
};

}  // namespace lite
}  // namespace paddle
