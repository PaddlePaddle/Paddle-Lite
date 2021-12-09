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

#ifndef LITE_BACKENDS_METAL_METAL_IMAGE_H_
#define LITE_BACKENDS_METAL_METAL_IMAGE_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <vector>

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_context.h"
#include "lite/core/dim.h"

namespace paddle {
namespace lite {

class MetalImage {
   public:
    MetalImage() = delete;
    virtual ~MetalImage();

    MetalImage(MetalContext* context,
        const DDim& in_dim,
        std::vector<int> in_transpose = {0, 2, 3, 1},
        METAL_PRECISION_TYPE precision_type = METAL_PRECISION_TYPE::HALF,
        METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite,
        bool use_mps = false);

    void initImage(MetalContext* context);

    void initImageReuse(MetalContext* context, std::string ptr);

    //  source tensor for mps
    void* src_tensor_{nullptr};

#if defined(__OBJC__)
    id<MTLTexture> image() const;
#endif

    int ElementCount() const;

    template <typename SP>
    void CopyFromNCHW(const SP* src);

    template <typename DP>
    void CopyToNCHW(DP* dst) const;

    static DDim FourDimFrom(DDim in_dim);
    __unused void Zero() const;

   public:
    bool use_mps_ = false;
    size_t channels_per_pixel_{};
    size_t array_length_{};
    size_t texture_width_{};
    size_t texture_height_{};

    DDim tensor_dim_;
    DDim dim_;
    DDim pad_to_four_dim_;
    std::vector<int> transpose_ = {0, 1, 2, 3};

   private:
    void InitTexture();
    const METAL_PRECISION_TYPE precision_type_;
    const METAL_ACCESS_FLAG flag_;

#if defined(__OBJC__)
    id<MTLTexture> image_{nil};
    MTLTextureDescriptor* desc_{nil};
    // memory reuse
    API_AVAILABLE(ios(10.0))
    id<MTLHeap> heap_{nullptr};
    API_AVAILABLE(ios(10.0))
    void initImageFromHeap(MetalContext* context, std::string ptr);
#endif
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_IMAGE_H_
