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

#include <array>
#include <vector>

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_device.h"
#include "lite/core/dim.h"

namespace paddle {
namespace lite {

class metal_image {
 public:
  metal_image() = delete;
  virtual ~metal_image();

#if defined(__OBJC__)
  id<MTLTexture> get_image() const;
  id<MTLTexture> mtl_image_{nil};
  id<MTLDevice> mtl_device_{nil};
  MTLTextureDescriptor* desc_{nil};
#else
  void* get_image() const;
  void* mtl_image_{nullptr};
  metal_device* mtl_device_;
  void* desc_{nullptr};
#endif

  metal_image(const metal_device& device,
              const DDim& inDim,
              std::vector<int> inTranspose = {0, 2, 3, 1},
              METAL_PRECISION_TYPE precision_type = METAL_PRECISION_TYPE::FLOAT,
              METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  template <typename SP>
  void from_nchw(const SP* src);

  template <typename P>
  void to_nchw(P* dst) const;

  static DDim fourDimFrom(DDim inDim);
  void zero() const;
  __unused void updateDim(DDim inDim);

  // std::recursive_mutex buffer_lock_;
  size_t size_{};
  bool useMPS_ = false;
  size_t channelsPerPixels_{};
  size_t arrayLength_{};
  size_t textureWidth_{};
  size_t textureHeight_{};

  DDim tensorDim_;
  DDim dim_;
  DDim padToFourDim_;
  std::vector<int> transpose_ = {0, 1, 2, 3};

 private:
  void updateDims(const DDim& inTensorDim);
  void initTexture(std::vector<int> inTranspose);
  const METAL_PRECISION_TYPE precisionType_;
  const METAL_ACCESS_FLAG flag_;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_IMAGE_H_
