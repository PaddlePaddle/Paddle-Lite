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


#include "lite/backends/metal/target_wrapper.h"
#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_image.h"
#include <cassert>

namespace paddle {
namespace lite {


size_t TargetWrapperMetal::num_devices() {
    return ctx_.GetDevicesNum();
}

void* TargetWrapperMetal::Malloc(size_t size) {
    return nullptr;
}

void TargetWrapperMetal::WaitForCompleted() {
    auto backend = (__bridge MetalContextImp*)ctx_.backend();
    [backend waitUntilCompleted];
}

template <>
void* TargetWrapperMetal::MallocImage<float>(const DDim dim,
                                             std::vector<int> transpose,
                                             void* host_ptr) {
    auto image = new MetalImage(&ctx_, dim, transpose, METAL_PRECISION_TYPE::FLOAT);
    if (host_ptr) image->CopyFromNCHW<float>((float*)host_ptr);
    return (void*)image;
}

template <>
void* TargetWrapperMetal::MallocImage<MetalHalf>(const DDim dim,
                                                 std::vector<int> transpose,
                                                 void* host_ptr) {
    auto image = new MetalImage(&ctx_, dim, transpose, METAL_PRECISION_TYPE::HALF);
    if (host_ptr) image->CopyFromNCHW<MetalHalf>((MetalHalf*)host_ptr);
    return (void*)image;
}

void* TargetWrapperMetal::MallocBuffer(size_t size, METAL_ACCESS_FLAG access) {
    return nullptr;
}

void TargetWrapperMetal::FreeImage(void* image) {
    if (image != nullptr) {
        delete (MetalImage*)image;
        image = nullptr;
    }
}

void TargetWrapperMetal::Free(void* ptr) {
    if (ptr != nullptr) {
        delete (MetalBuffer*)ptr;
        ptr = nullptr;
    }
    return;
}

void TargetWrapperMetal::MemcpySync(void* dst, const void* src, size_t size, IoDirection dir) {
    switch (dir) {
        case IoDirection::DtoD: {
            auto dst_buffer = (MetalBuffer*)dst;
            auto src_buffer = (MetalBuffer*)src;
            dst_buffer->Copy(reinterpret_cast<const MetalBuffer&>(src_buffer), size, 0, 0);
            break;
        }
        case IoDirection::HtoD: {
            auto dst_buffer = (MetalBuffer*)dst;
            dst_buffer->Read(const_cast<void*>(src), size, 0);
            break;
        }
        case IoDirection::DtoH: {
            auto src_buffer = (MetalBuffer*)src;
            src_buffer->Write(const_cast<void*>(dst), size, 0);
            break;
        }
        default:
            LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
    }
}

void TargetWrapperMetal::MemsetSync(void* devPtr, int value, size_t count) {
    return;
}

LITE_THREAD_LOCAL MetalContext TargetWrapperMetal::ctx_;

}  // namespace lite
}  // namespace paddle
