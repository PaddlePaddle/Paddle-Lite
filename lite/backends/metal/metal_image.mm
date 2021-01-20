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


#include "lite/backends/metal/metal_image.h"
#include "lite/backends/metal/metal_half.h"

namespace paddle {
namespace lite {

static MTLResourceOptions option_for_access(METAL_ACCESS_FLAG flag) {
  if (flag == METAL_ACCESS_FLAG::CPUWriteOnly) {
    return MTLResourceOptionCPUCacheModeWriteCombined;
  } else if (flag == METAL_ACCESS_FLAG::CPUTransparent) {
    if (@available(iOS 9.0, *)) {
      return MTLResourceStorageModePrivate;
    } else {
      return MTLResourceOptionCPUCacheModeDefault;
    }
  } else {
    return MTLResourceOptionCPUCacheModeDefault;
  }
}

metal_image::metal_image(const metal_device &device,
                         const DDim &inDim,
                         std::vector<int> inTranspose,
                         const METAL_PRECISION_TYPE precision_type,
                         const METAL_ACCESS_FLAG flag)
    : precisionType_(precision_type), flag_(flag) {
  mtl_device_ = device.get_device();
  auto fourDim = fourDimFrom(inDim);

  tensorDim_ = inDim;
  dim_ = fourDim;
  padToFourDim_ = fourDim;

  if (tensorDim_.size() < 4) {
    transpose_ = {0, 1, 2, 3};
  } else {
    transpose_ = std::move(inTranspose);
  }

  initTexture(transpose_);

  mtl_image_ = [mtl_device_ newTextureWithDescriptor:desc_];
  // TODO: Do we need clear the buffer here
}

__unused void metal_image::updateDim(DDim inDim) {
  auto h = 0;
  auto w = 0;
  std::vector<int> nhwc{0, 2, 3, 1};
  std::vector<int> nchw{0, 1, 2, 3};
  if (transpose_ == nhwc) {
    h = static_cast<int>(inDim[1]);
    w = static_cast<int>(inDim[2]);
  } else if (transpose_ == nchw) {
    assert(inDim.size() == 4);
    h = static_cast<int>(inDim[2]);
    w = static_cast<int>(inDim[3]);
  } else {
    throw std::logic_error("ERROR: unsupported transpose");
  }
  DDim newTensorDim;
  if (tensorDim_.size() == 4) {
    newTensorDim = DDimLite({tensorDim_[0], tensorDim_[1], h, w});
  } else if (tensorDim_.size() == 3) {
    newTensorDim = DDimLite({tensorDim_[0], h, w});
  } else if (tensorDim_.size() == 2) {
    newTensorDim = DDimLite({h, w});
  } else {
    throw std::logic_error("ERROR: unsupported tensor dim count");
  }
  updateDims(newTensorDim);
  initTexture(transpose_);
}

void metal_image::updateDims(const DDim &inTensorDim) {
  auto fourDim = metal_image::fourDimFrom(inTensorDim);
  tensorDim_ = inTensorDim;
  padToFourDim_ = fourDim;

  std::vector<DDim::value_type> newDim = {};
  for_each(transpose_.begin(), transpose_.end(), [&](int i) -> void {
    newDim.emplace_back(padToFourDim_[i]);
  });
  dim_ = DDimLite(newDim);
}

void metal_image::initTexture(std::vector<int> inTranspose) {
  transpose_ = std::move(inTranspose);
  std::vector<DDim::value_type> newDim = {};
  for_each(transpose_.begin(), transpose_.end(), [&](int i) -> void {
    newDim.emplace_back(padToFourDim_[i]);
  });

  dim_ = DDimLite(newDim);

  desc_ = [[MTLTextureDescriptor alloc] init];
  [desc_ setTextureType:MTLTextureType2DArray];
  [desc_ setDepth:1];

  switch (tensorDim_.size()) {
    case 4:
      desc_.width = static_cast<NSUInteger>(newDim[2]);
      desc_.height = static_cast<NSUInteger>(newDim[1]);
      desc_.arrayLength = static_cast<NSUInteger>(((newDim[0]) * (newDim[3]) + 3) / 4);
      break;
    case 3:
      desc_.width = static_cast<NSUInteger>(newDim[3]);
      desc_.height = static_cast<NSUInteger>(newDim[2]);
      desc_.arrayLength = static_cast<NSUInteger>((newDim[1] + 3) / 4);
      break;
    case 2:
    case 1:
      desc_.width = static_cast<NSUInteger>((newDim[3] + 3) / 4);
      desc_.height = static_cast<NSUInteger>(newDim[2]);
      desc_.arrayLength = 1;
      break;
    default:
      throw std::logic_error("ERROR: Dim size is error");
  }

  textureWidth_ = desc_.width;
  textureHeight_ = desc_.height;
  arrayLength_ = desc_.arrayLength;

  if (precisionType_ == METAL_PRECISION_TYPE::HALF) {
    if (useMPS_) {
      if (tensorDim_[1] == 1) {
        desc_.pixelFormat = MTLPixelFormatR16Float;
        channelsPerPixels_ = 1;
      } else {
        desc_.pixelFormat = MTLPixelFormatRGBA16Float;
        channelsPerPixels_ = 4;
      }
    } else {
      desc_.pixelFormat = MTLPixelFormatRGBA16Float;
      channelsPerPixels_ = 4;
    }
  } else if (precisionType_ == METAL_PRECISION_TYPE::FLOAT) {
    if (useMPS_) {
      if (tensorDim_[1] == 1) {
        desc_.pixelFormat = MTLPixelFormatR32Float;
        channelsPerPixels_ = 1;
      } else {
        desc_.pixelFormat = MTLPixelFormatRGBA32Float;
        channelsPerPixels_ = 4;
      }
    } else {
      desc_.pixelFormat = MTLPixelFormatRGBA32Float;
      channelsPerPixels_ = 4;
    }
  }

  desc_.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc_.storageMode = MTLStorageModeShared;
  mtl_image_ = nil;
}

DDim metal_image::fourDimFrom(DDim inDim) {
  DDim fourDim;
  auto inDimVec = inDim.Vectorize();
  if (inDim.size() == 4) {
    fourDim = inDim;
  } else if (inDim.size() < 4) {
    std::vector<DDim::value_type> fourDimNum = {static_cast<DDim::value_type>(1),
                                                static_cast<DDim::value_type>(1),
                                                static_cast<DDim::value_type>(1),
                                                static_cast<DDim::value_type>(1)};

    for (int i = 0; i < inDim.size(); ++i) {
      fourDimNum[4 - inDim.size() + i] = inDim[i];
    }
    fourDim = DDimLite(fourDimNum);
  } else {
    throw std::logic_error("Error: cannot support such dims");
  }
  return fourDim;
}

#if defined(__OBJC__)
id<MTLTexture> metal_image::get_image() const
#else
void *metal_image::get_image() const
#endif
{
  return mtl_image_;
}

template <typename SP>
void metal_image::from_nchw(const SP *src) {
  size_t newDims[] = {1, 1, 1, 1};
  for (int i = 0; i < tensorDim_.size(); ++i) {
    newDims[4 - tensorDim_.size() + i] = static_cast<size_t>(tensorDim_[i]);
  }

  size_t N = newDims[0];
  size_t C = newDims[1];
  size_t H = newDims[2];
  size_t W = newDims[3];

  auto rcount = textureWidth_ * textureHeight_ * arrayLength_ * channelsPerPixels_;

  if (precisionType_ == METAL_PRECISION_TYPE::FLOAT && std::is_same<SP, float>::value) {
    auto nvalue = (float *)malloc(sizeof(float) * rcount);
    if (tensorDim_.size() > 2) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < H; ++i1) {
          for (int i2 = 0; i2 < W; ++i2) {
            for (int i3 = 0; i3 < C; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i3 * H * W) + (i1 * W) + i2;
              auto k = ig[0] * C + ig[3];
              auto jx = ((k / 4) * H * W * 4) + (ig[1] * W * 4) + (ig[2] * 4) + (k % 4);
              nvalue[jx] = src[ix];
            }
          }
        }
      }
    } else {
      for (int i1 = 0; i1 < H; ++i1) {
        for (int i2 = 0; i2 < W; ++i2) {
          auto ix = (i1 * W) + i2;
          auto jx = (i1 * textureWidth_ * 4) + i2;
          nvalue[jx] = src[ix];
        }
      }
    }

    auto bytesPerRow = mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(float);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};

    for (int i = 0; i < arrayLength_; ++i) {
      auto p = nvalue + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;
      [mtl_image_ replaceRegion:region
                    mipmapLevel:0
                          slice:static_cast<NSUInteger>(i)
                      withBytes:p
                    bytesPerRow:bytesPerRow
                  bytesPerImage:bytesPerImage];
    }

    free(nvalue);
  } else if (precisionType_ == METAL_PRECISION_TYPE::HALF && std::is_same<SP, float>::value) {
    auto nvalue = (metal_half *)malloc(sizeof(metal_half) * rcount);
    if (tensorDim_.size() > 2) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < H; ++i1) {
          for (int i2 = 0; i2 < W; ++i2) {
            for (int i3 = 0; i3 < C; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i3 * H * W) + (i1 * W) + i2;
              auto k = ig[0] * C + ig[3];
              auto jx = ((k / 4) * H * W * 4) + (ig[1] * W * 4) + (ig[2] * 4) + (k % 4);
              nvalue[jx] = MetalFloat2Half(src[ix]);
            }
          }
        }
      }
    } else {
      for (int i1 = 0; i1 < H; ++i1) {
        for (int i2 = 0; i2 < W; ++i2) {
          auto ix = (i1 * W) + i2;
          auto jx = (i1 * textureWidth_ * 4) + i2;
          nvalue[jx] = MetalFloat2Half(src[ix]);
        }
      }
    }

    auto bytesPerRow =
        mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(metal_half);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};

    for (int i = 0; i < arrayLength_; ++i) {
      auto p = nvalue + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;
      [mtl_image_ replaceRegion:region
                    mipmapLevel:0
                          slice:static_cast<NSUInteger>(i)
                      withBytes:p
                    bytesPerRow:bytesPerRow
                  bytesPerImage:bytesPerImage];
    }

    free(nvalue);
  } else if (precisionType_ == METAL_PRECISION_TYPE::HALF && std::is_same<SP, metal_half>::value) {
    auto nvalue = (metal_half *)malloc(sizeof(metal_half) * rcount);
    if (tensorDim_.size() > 2) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < H; ++i1) {
          for (int i2 = 0; i2 < W; ++i2) {
            for (int i3 = 0; i3 < C; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i3 * H * W) + (i1 * W) + i2;
              auto k = ig[0] * C + ig[3];
              auto jx = ((k / 4) * H * W * 4) + (ig[1] * W * 4) + (ig[2] * 4) + (k % 4);
              nvalue[jx] = (src[ix]);
            }
          }
        }
      }
    } else {
      for (int i1 = 0; i1 < H; ++i1) {
        for (int i2 = 0; i2 < W; ++i2) {
          auto ix = (i1 * W) + i2;
          auto jx = (i1 * textureWidth_ * 4) + i2;
          nvalue[jx] = (src[ix]);
        }
      }
    }

    auto bytesPerRow =
        mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(metal_half);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};

    for (int i = 0; i < arrayLength_; ++i) {
      auto p = nvalue + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;
      [mtl_image_ replaceRegion:region
                    mipmapLevel:0
                          slice:static_cast<NSUInteger>(i)
                      withBytes:p
                    bytesPerRow:bytesPerRow
                  bytesPerImage:bytesPerImage];
    }

    free(nvalue);
  } else {
    throw std::logic_error("ERROR: do not support this half format");
    return;
  }
}

void metal_image::zero() const {
  if (mtl_image_ == nullptr) return;

  int size_p = 1;
  if (precisionType_ == METAL_PRECISION_TYPE::FLOAT)
    size_p = 4;
  else if (precisionType_ == METAL_PRECISION_TYPE::HALF)
    size_p = 2;
  auto rcount = textureWidth_ * textureHeight_ * 1 * channelsPerPixels_;
  char *nvalue = (char *)malloc(size_p * rcount);
  memset(nvalue, 0, size_p * rcount);

  auto bytesPerRow = mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * size_p;
  auto bytesPerImage = mtl_image_.height * bytesPerRow;
  const MTLRegion region{.origin = {0, 0, 0},
                         .size = {
                             mtl_image_.width,
                             mtl_image_.height,
                             mtl_image_.depth,
                         }};

  {
    int i = static_cast<int>(arrayLength_ - 1);
    auto p = nvalue;
    [mtl_image_ replaceRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(i)
                    withBytes:p
                  bytesPerRow:bytesPerRow
                bytesPerImage:bytesPerImage];
  }

  free(nvalue);
}

template <typename P>
void metal_image::to_nchw(P *dst) const {
  size_t newDims[] = {1, 1, 1, 1};
  for (int i = 0; i < tensorDim_.size(); ++i) {
    newDims[4 - tensorDim_.size() + i] = static_cast<size_t>(tensorDim_[i]);
  }

  size_t N = newDims[0];
  size_t C = newDims[1];
  size_t H = newDims[2];
  size_t W = newDims[3];

  auto dstCounts = arrayLength_ * channelsPerPixels_ * textureWidth_ * textureHeight_;

  if (precisionType_ == METAL_PRECISION_TYPE::FLOAT && std::is_same<P, float>::value) {
    auto pointer = (float *)malloc(sizeof(float) * dstCounts);

    auto bytesPerRow = mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(float);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};
    for (int i = 0; i < arrayLength_; ++i) {
      auto p = pointer + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;

      [mtl_image_ getBytes:(p)
               bytesPerRow:(bytesPerRow)bytesPerImage:(bytesPerImage)fromRegion:(region)mipmapLevel
                          :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensorDim_.size() > 2) {
      for (int s = 0; s < arrayLength_; s++) {
        for (int c = 0; c < 4; c++) {
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              if ((s * 4 + c) < (C * N)) {
                dst[index++] = pointer[W * H * 4 * s + h * W * 4 + w * 4 + c];
              }
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = pointer[textureWidth_ * 4 * h + w];
        }
      }
    }
    free(pointer);
  } else if (precisionType_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, float>::value) {
    auto pointer = (metal_half *)malloc(sizeof(metal_half) * dstCounts);

    auto bytesPerRow =
        mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(metal_half);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};
    for (int i = 0; i < arrayLength_; ++i) {
      auto p = pointer + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;

      [mtl_image_ getBytes:(p)
               bytesPerRow:(bytesPerRow)bytesPerImage:(bytesPerImage)fromRegion:(region)mipmapLevel
                          :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensorDim_.size() > 2) {
      for (int s = 0; s < arrayLength_; s++) {
        for (int c = 0; c < 4; c++) {
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              if ((s * 4 + c) < (C * N)) {
                dst[index++] = MetalHalf2Float(pointer[W * H * 4 * s + h * W * 4 + w * 4 + c]);
              }
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = MetalHalf2Float(pointer[textureWidth_ * 4 * h + w]);
        }
      }
    }
    free(pointer);
  } else if (precisionType_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, metal_half>::value) {
    auto pointer = (metal_half *)malloc(sizeof(metal_half) * dstCounts);

    auto bytesPerRow =
        mtl_image_.width * mtl_image_.depth * channelsPerPixels_ * sizeof(metal_half);
    auto bytesPerImage = mtl_image_.height * bytesPerRow;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               mtl_image_.width,
                               mtl_image_.height,
                               mtl_image_.depth,
                           }};
    for (int i = 0; i < arrayLength_; ++i) {
      auto p = pointer + mtl_image_.width * mtl_image_.height * channelsPerPixels_ * i;

      [mtl_image_ getBytes:(p)
               bytesPerRow:(bytesPerRow)bytesPerImage:(bytesPerImage)fromRegion:(region)mipmapLevel
                          :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensorDim_.size() > 2) {
      for (int s = 0; s < arrayLength_; s++) {
        for (int c = 0; c < 4; c++) {
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              if ((s * 4 + c) < (C * N)) {
                dst[index++] = pointer[W * H * 4 * s + h * W * 4 + w * 4 + c];
              }
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = pointer[textureWidth_ * 4 * h + w];
        }
      }
    }
    free(pointer);
  }
}

metal_image::~metal_image() {
  mtl_image_ = nil;
  mtl_device_ = nil;
  if (desc_ != nil) {
    [desc_ release];
    desc_ = nil;
  }
}

template void metal_image::from_nchw(const float *src);
template void metal_image::from_nchw(const metal_half *src);
template void metal_image::to_nchw(float *dst) const;
template void metal_image::to_nchw(metal_half *dst) const;

}
}