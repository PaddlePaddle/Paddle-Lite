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

#include "lite/backends/metal/target_wrapper.h"
#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/backends/metal/metal_mtl_data.h"
#include <cassert>

namespace paddle {
namespace lite {

template <>
void* TargetWrapperMetal::MallocImage<float>(MetalContext* context,
    const DDim dim,
    std::vector<int> transpose) {
    auto image = new MetalImage(context, dim, transpose, METAL_PRECISION_TYPE::FLOAT);
    return (void*)image;
}

template <>
void* TargetWrapperMetal::MallocImage<MetalHalf>(MetalContext* context,
    const DDim dim,
    std::vector<int> transpose) {
    auto image = new MetalImage(context, dim, transpose, METAL_PRECISION_TYPE::HALF);
    return (void*)image;
}

void TargetWrapperMetal::FreeImage(void* image) {
    if (image != nullptr) {
        delete (MetalImage*)image;
        image = nullptr;
    }
}

void* TargetWrapperMetal::MallocMTLData(void* ptr) {
    auto texture = new MetalMTLData(ptr);
    return (void*)texture;
}

void TargetWrapperMetal::FreeMTLData(void* ptr) {
    if (ptr != nullptr) {
        delete (MetalMTLData*)ptr;
        ptr = nullptr;
    }
}

void* TargetWrapperMetal::Malloc(size_t size) {
    return malloc(size);
}

void TargetWrapperMetal::Free(void* ptr) {
    if (ptr) {
        free(ptr);
        ptr = nullptr;
    }
}

void TargetWrapperMetal::MemcpySync(void* dst, const void* src, size_t size, IoDirection dir) {
    if (size > 0) {
        memcpy(dst, src, size);
    }
}

void TargetWrapperMetal::MemsetSync(void* dst, int value, size_t size) {
    if (size) {
        memset(dst, value, size);
    }
}

}  // namespace lite
}  // namespace paddle
