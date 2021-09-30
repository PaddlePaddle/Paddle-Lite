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

#ifndef LITE_BACKENDS_METAL_METAL_MTL_DATA_H_
#define LITE_BACKENDS_METAL_METAL_MTL_DATA_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

namespace paddle {
namespace lite {

class MetalMTLData {
   public:
    MetalMTLData() = delete;
    virtual ~MetalMTLData();

    MetalMTLData(void* ptr);
#if defined(__OBJC__)
    id<MTLTexture> image() const;
#endif

   private:
#if defined(__OBJC__)
    id<MTLTexture> image_{nil};
#endif
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_MTL_DATA_H_
