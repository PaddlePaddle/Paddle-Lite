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


#include <cassert>
#include "lite/core/tensor.h"
#include "lite/backends/metal/target_wrapper.h"

namespace paddle {
namespace lite {

metal_context global_ctx;
size_t TargetWrapperMetal::num_devices() { return global_ctx.get_devices_num(); }

void* TargetWrapperMetal::Malloc(size_t size) {
  void* ptr{};
  auto device = global_ctx.get_default_device();
  auto buffer = new metal_buffer(*device, size);
  return (void*)buffer;
}

template <>
void* TargetWrapperMetal::MallocImage<float>(const DDim dim,
                                             std::vector<int> transpose,
                                             void* host_ptr) {
  void* ptr{};
  auto device = global_ctx.get_default_device();
  auto image = new metal_image(*device, dim, transpose, METAL_PRECISION_TYPE::FLOAT);
  if (host_ptr) image->from_nchw<float>((float*)host_ptr);
  return (void*)image;
}

template <>
void* TargetWrapperMetal::MallocImage<metal_half>(const DDim dim,
                                                  std::vector<int> transpose,
                                                  void* host_ptr) {
  void* ptr{};
  auto device = global_ctx.get_default_device();
  auto image = new metal_image(*device, dim, transpose, METAL_PRECISION_TYPE::HALF);
  if (host_ptr) image->from_nchw<metal_half>((metal_half*)host_ptr);
  return (void*)image;
}

void TargetWrapperMetal::FreeImage(void* image) {
  if (image != nullptr) {
    delete (metal_image*)image;
    image = nullptr;
  }
}

void TargetWrapperMetal::Free(void* ptr) {
  if (ptr != NULL) {
    delete (metal_buffer*)ptr;
    ptr = NULL;
  }
  return;
}

void TargetWrapperMetal::MemcpySync(void* dst, const void* src, size_t size, IoDirection dir) {
  switch (dir) {
    case IoDirection::DtoD: {
      auto dst_buffer = (metal_buffer*)dst;
      auto src_buffer = (metal_buffer*)src;
      dst_buffer->copy(reinterpret_cast<const metal_buffer&>(src_buffer), size, 0, 0);
      break;
    }
    case IoDirection::HtoD: {
      auto dst_buffer = (metal_buffer*)dst;
      dst_buffer->read(const_cast<void*>(src), size, 0);
      break;
    }
    case IoDirection::DtoH: {
      auto src_buffer = (metal_buffer*)src;
      src_buffer->write(const_cast<void*>(dst), size, 0);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

void TargetWrapperMetal::MemsetSync(void* devPtr, int value, size_t count) { return; }

}  // namespace lite
}  // namespace paddle
