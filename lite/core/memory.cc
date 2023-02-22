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

#include "lite/core/memory.h"

#ifdef LITE_WITH_METAL
#include "lite/backends/metal/target_wrapper.h"
#include "lite/core/device_info.h"
#endif

namespace paddle {
namespace lite {

void* TargetMalloc(TargetType target, size_t size) {
  void* data{nullptr};
  if (lite::Allocator::Global().GetCustomAllocator().alloc) {
    data = lite::Allocator::Global().GetCustomAllocator().alloc(
        size, host::MALLOC_ALIGN);
  } else {
    switch (target) {
      case TargetType::kHost:
      case TargetType::kX86:
      case TargetType::kARM:
        data = TargetWrapper<TARGET(kHost)>::Malloc(size);
        break;
#ifdef LITE_WITH_OPENCL
      case TargetType::kOpenCL:
        data = TargetWrapperCL::Malloc(size);
        break;
#endif  // LITE_WITH_OPENCL
#ifdef LITE_WITH_XPU
      case TargetType::kXPU:
        data = TargetWrapperXPU::Malloc(size);
        break;
#endif  // LITE_WITH_XPU
#ifdef LITE_WITH_METAL
      case TargetType::kMetal: {
        data = TargetWrapperMetal::Malloc(size);
        break;
      }
#endif  // LITE_WITH_METAL
      default:
        LOG(FATAL) << "Unknown supported target " << TargetToStr(target);
    }
  }
  return data;
}

void TargetFree(TargetType target, void* data, std::string free_flag) {
  if (lite::Allocator::Global().GetCustomAllocator().free) {
    lite::Allocator::Global().GetCustomAllocator().free(data);
  } else {
    switch (target) {
      case TargetType::kHost:
      case TargetType::kX86:
      case TargetType::kARM:
        TargetWrapper<TARGET(kHost)>::Free(data);
        break;

#ifdef LITE_WITH_OPENCL
      case TargetType::kOpenCL:
        if (free_flag == "cl_use_image2d_") {
          TargetWrapperCL::FreeImage(data);
        } else {
          TargetWrapperCL::Free(data);
        }
        break;
#endif  // LITE_WITH_OPENCL
#ifdef LITE_WITH_XPU
      case TargetType::kXPU:
        TargetWrapperXPU::Free(data);
        break;
#endif  // LITE_WITH_XPU
#ifdef LITE_WITH_METAL
      case TargetType::kMetal:
        if (free_flag == "metal_use_image2d_") {
          TargetWrapperMetal::FreeImage(data);
        } else {
          TargetWrapperMetal::Free(data);
        }
        break;
#endif
      default:
        LOG(FATAL) << "Unknown supported target:" << TargetToStr(target);
    }
  }
}

void TargetCopy(TargetType target, void* dst, const void* src, size_t size) {
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      TargetWrapper<TARGET(kHost)>::MemcpySync(
          dst, src, size, IoDirection::DtoD);
      break;

#ifdef LITE_WITH_XPU
    case TargetType::kXPU:
      TargetWrapperXPU::MemcpySync(dst, src, size, IoDirection::HtoD);
      break;
#endif
#ifdef LITE_WITH_OPENCL
    case TargetType::kOpenCL:
      TargetWrapperCL::MemcpySync(dst, src, size, IoDirection::DtoD);
      break;
#endif  // LITE_WITH_OPENCL
#ifdef LITE_WITH_Metal
    case TargetType::kMetal:
      TargetWrapperMetal::MemcpySync(dst, src, size, IoDirection::DtoD);
      break;
#endif
    default:
      LOG(FATAL) << "Unknown supported target:" << TargetToStr(target);
  }
}

#ifdef LITE_WITH_OPENCL
void TargetCopyImage2D(TargetType target,
                       void* dst,
                       const void* src,
                       const size_t cl_image2d_width,
                       const size_t cl_image2d_height,
                       const size_t cl_image2d_row_pitch,
                       const size_t cl_image2d_slice_pitch) {
  TargetWrapperCL::ImgcpySync(dst,
                              src,
                              cl_image2d_width,
                              cl_image2d_height,
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoD);
}
#endif

}  // namespace lite
}  // namespace paddle
