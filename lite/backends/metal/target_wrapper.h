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

#pragma once

#include <string>
#include <vector>

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_context.h"
#include "lite/core/dim.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperMetal = TargetWrapper<TARGET(kMetal)>;

template <>
class TargetWrapper<TARGET(kMetal)> {
   public:
    template <typename T>
    static void* MallocImage(MetalContext* context, const DDim dim, std::vector<int> transport);

    static void FreeImage(void* image);

    static void* MallocMTLData(void* ptr);

    static void FreeMTLData(void* ptr);

    static void* Malloc(size_t size);

    static void Free(void* ptr);

    static void MemcpySync(void* dst,
        const void* src,
        size_t size,
        IoDirection dir = lite::IoDirection::HtoH);

    static void MemsetSync(void* dst, int value, size_t size);
};
}  // namespace lite
}  // namespace paddle
