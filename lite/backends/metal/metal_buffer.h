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

#ifndef LITE_BACKENDS_METAL_METAL_BUFFER_H_
#define LITE_BACKENDS_METAL_METAL_BUFFER_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <algorithm>

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_context.h"
#include "lite/backends/metal/metal_converter.h"

namespace paddle {
namespace lite {

/**
 * Data arrangement :
 * rawdata is NCHW
 * MTLBuffer is NHWC
 */
class MetalBuffer {
   public:
    // scenes: metal shader params.
    // only contains MTLBuffer eg: conv2d params
    MetalBuffer(MetalContext* context,
        size_t size,
        void* data = nullptr,
        METAL_ACCESS_FLAG access = METAL_ACCESS_FLAG::CPUWriteOnly);

    // scenes: shader input or output
    // only contains MTLBuffer eg: io_copy input or fetch output
    MetalBuffer(MetalContext* context,
        const DDim& inDim,
        size_t size,
        void* data = nullptr,
        METAL_PRECISION_TYPE precision = METAL_PRECISION_TYPE::FLOAT,
        METAL_ACCESS_FLAG access = METAL_ACCESS_FLAG::CPUShared);

    // scenes: shader weight
    // contains raw_data_ and MTLBuffer eg: conv2d filter
    MetalBuffer(MetalContext* context,
        const DDim& inDim,
        METAL_PRECISION_TYPE precision = METAL_PRECISION_TYPE::HALF);

    MetalBuffer() = delete;
    ~MetalBuffer();

    bool convert_to_nhwc_{true};
    bool with_transpose_{false};
    bool pad_when_one_channel_{false};
    DataLayout data_layout_{DataLayout::kNCHW};  // raw data layout
    METAL_ACCESS_FLAG access_{METAL_ACCESS_FLAG::CPUWriteOnly};
    METAL_PRECISION_TYPE precision_{METAL_PRECISION_TYPE::HALF};

#if defined(__OBJC__)
    id<MTLBuffer> buffer() const;
#endif
    void* rawdata() {
        return rawdata_;
    }
    DDim tensor_dim() {
        return tensor_dim_;
    }
    size_t mtl_size() const {
        return mtl_size_;
    }

   public:
    template <typename SP>
    void CopyFromNCHW(const SP* src);

    template <typename DP>
    void CopyToNCHW(DP* dst);

    template <typename P>
    P* Convert(DataConverter<P>* converter);

   private:
    DDim dim_;
    DDim tensor_dim_;
    MetalContext* metal_context_;

    void* rawdata_ = nullptr;  // cpu raw data NCHW
    size_t mtl_size_;          // MTLBuffer byte size
    size_t precision_size_;
    bool can_copy_to_{true};  // is can copy to someone

#if defined(__OBJC__)
    id<MTLBuffer> buffer_{nil};  // NHWC GPU data
#endif

    void Convert2NHWC();  // NCHW to NHWC
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_BUFFER_H_
