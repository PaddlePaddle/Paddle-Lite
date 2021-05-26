#import <__bit_reference>
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

#include "lite/backends/metal/metal_half.h"
#include "lite/backends/metal/metal_image.h"

namespace paddle {
namespace lite {

static MTLResourceOptions OptionForAccess(METAL_ACCESS_FLAG flag) {
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

MetalImage::MetalImage(const MetalDevice &device,
                       const DDim &in_dim,
                       std::vector<int> in_transpose,
                       const METAL_PRECISION_TYPE precision_type,
                       const METAL_ACCESS_FLAG flag)
    : precision_type_(precision_type), flag_(flag) {
  device_ = &device;
  auto four_dim = FourDimFrom(in_dim);

  tensor_dim_ = in_dim;
  dim_ = four_dim;
  pad_to_four_dim_ = four_dim;

  if (tensor_dim_.size() < 4) {
    transpose_ = {0, 1, 2, 3};
  } else {
    transpose_ = std::move(in_transpose);
  }

  UpdateDims(in_dim);
  InitTexture();

  // TODO:(lzy) Do we need clear the buffer here
}

void MetalImage::UpdateDims(const DDim &in_tensor_dim) {
  auto four_dim = MetalImage::FourDimFrom(in_tensor_dim);
  tensor_dim_ = in_tensor_dim;
  pad_to_four_dim_ = four_dim;

  std::vector<DDim::value_type> new_dim = {};
  for_each(transpose_.begin(), transpose_.end(), [&](int i) -> void {
    new_dim.emplace_back(pad_to_four_dim_[i]);
  });
  dim_ = DDimLite(new_dim);
}

void MetalImage::InitTexture() {

  std::vector<DDim::value_type> new_dim = {};
  for_each(transpose_.begin(), transpose_.end(), [&](int i) -> void {
    new_dim.emplace_back(pad_to_four_dim_[i]);
  });

  dim_ = DDimLite(new_dim);

  desc_ = [[MTLTextureDescriptor alloc] init];
  [desc_ setTextureType:MTLTextureType2DArray];
  [desc_ setDepth:1];

  switch (tensor_dim_.size()) {
    case 4:
      desc_.width = static_cast<NSUInteger>(new_dim[2]);
      desc_.height = static_cast<NSUInteger>(new_dim[1]);
      desc_.arrayLength = static_cast<NSUInteger>(((new_dim[0]) * (new_dim[3]) + 3) / 4);
      break;
    case 3:
      desc_.width = static_cast<NSUInteger>(new_dim[3]);
      desc_.height = static_cast<NSUInteger>(new_dim[2]);
      desc_.arrayLength = static_cast<NSUInteger>((new_dim[1] + 3) / 4);
      break;
    case 2:
    case 1:
      desc_.width = static_cast<NSUInteger>((new_dim[3] + 3) / 4);
      desc_.height = static_cast<NSUInteger>(new_dim[2]);
      desc_.arrayLength = 1;
      break;
    default:
      throw std::logic_error("ERROR: Dim size is error");
  }

  texture_width_ = desc_.width;
  texture_height_ = desc_.height;
  array_length_ = desc_.arrayLength;

  if (precision_type_ == METAL_PRECISION_TYPE::HALF) {
    if (use_mps_) {
      if (tensor_dim_[1] == 1) {
        desc_.pixelFormat = MTLPixelFormatR16Float;
        channels_per_pixel_ = 1;
      } else {
        desc_.pixelFormat = MTLPixelFormatRGBA16Float;
        channels_per_pixel_ = 4;
      }
    } else {
      desc_.pixelFormat = MTLPixelFormatRGBA16Float;
      channels_per_pixel_ = 4;
    }
  } else if (precision_type_ == METAL_PRECISION_TYPE::FLOAT) {
    if (use_mps_) {
      if (tensor_dim_[1] == 1) {
        desc_.pixelFormat = MTLPixelFormatR32Float;
        channels_per_pixel_ = 1;
      } else {
        desc_.pixelFormat = MTLPixelFormatRGBA32Float;
        channels_per_pixel_ = 4;
      }
    } else {
      desc_.pixelFormat = MTLPixelFormatRGBA32Float;
      channels_per_pixel_ = 4;
    }
  }

  desc_.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
//  desc_.storageMode = MTLStorageModeShared;

  image_ = [device_->device() newTextureWithDescriptor:desc_];

#if (!__has_feature(objc_arc)) 
  [desc_ release];
#endif

  desc_ = nil;
}

DDim MetalImage::FourDimFrom(DDim in_dim) {
  DDim four_dim;
  if (in_dim.size() == 4) {
    four_dim = in_dim;
  } else if (in_dim.size() < 4) {
    std::vector<DDim::value_type> four_dim_num = {static_cast<DDim::value_type>(1),
                                                  static_cast<DDim::value_type>(1),
                                                  static_cast<DDim::value_type>(1),
                                                  static_cast<DDim::value_type>(1)};

    for (int i = 0; i < in_dim.size(); ++i) {
      four_dim_num[4 - in_dim.size() + i] = in_dim[i];
    }
    four_dim = DDimLite(four_dim_num);
  } else {
    throw std::logic_error("Error: cannot support such dims");
  }
  return four_dim;
}

#if defined(__OBJC__)
id<MTLTexture> MetalImage::image() const
#else
void *metal_image::image() const
#endif
{
  return image_;
}

template <typename SP>
void MetalImage::CopyFromNCHW(const SP *src) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (int i = 0; i < tensor_dim_.size(); ++i) {
    new_dims[4 - tensor_dim_.size() + i] = static_cast<size_t>(tensor_dim_[i]);
  }

  size_t N = new_dims[0];
  size_t C = new_dims[1];
  size_t H = new_dims[2];
  size_t W = new_dims[3];

  auto rcount = texture_width_ * texture_height_ * array_length_ * channels_per_pixel_;

  if (precision_type_ == METAL_PRECISION_TYPE::FLOAT && std::is_same<SP, float>::value) {
    auto nvalue = (float *)malloc(sizeof(float) * rcount);
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              nvalue[jx] = src[ix];
            }
          }
        }
      }
    } else if (tensor_dim_.size() == 3) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[1];
              auto jx =
                  ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
              nvalue[jx] = src[ix];
            }
          }
        }
      }
    } else {
      for (int i1 = 0; i1 < H; ++i1) {
        for (int i2 = 0; i2 < W; ++i2) {
          auto ix = (i1 * W) + i2;
          auto jx = (i1 * texture_width_ * 4) + i2;
          nvalue[jx] = src[ix];
        }
      }
    }

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(float);
    auto bytes_per_image = image_.height * bytes_per_row;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};

    for (int i = 0; i < array_length_; ++i) {
      auto p = nvalue + image_.width * image_.height * channels_per_pixel_ * i;
      [image_ replaceRegion:region
                mipmapLevel:0
                      slice:static_cast<NSUInteger>(i)
                  withBytes:p
                bytesPerRow:bytes_per_row
              bytesPerImage:bytes_per_image];
    }

    free(nvalue);
  } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<SP, float>::value) {
    auto nvalue = (MetalHalf *)malloc(sizeof(MetalHalf) * rcount);
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              nvalue[jx] = MetalFloat2Half(src[ix]);
            }
          }
        }
      }
    } else if (tensor_dim_.size() == 3) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[1];
              auto jx =
                  ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
              nvalue[jx] = MetalFloat2Half(src[ix]);
            }
          }
        }
      }
    } else {
      for (int i1 = 0; i1 < H; ++i1) {
        for (int i2 = 0; i2 < W; ++i2) {
          auto ix = (i1 * W) + i2;
          auto jx = (i1 * texture_width_ * 4) + i2;
          nvalue[jx] = MetalFloat2Half(src[ix]);
        }
      }
    }

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
    auto bytes_per_image = image_.height * bytes_per_row;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};

    for (int i = 0; i < array_length_; ++i) {
      auto p = nvalue + image_.width * image_.height * channels_per_pixel_ * i;
      [image_ replaceRegion:region
                mipmapLevel:0
                      slice:static_cast<NSUInteger>(i)
                  withBytes:p
                bytesPerRow:bytes_per_row
              bytesPerImage:bytes_per_image];
    }

    free(nvalue);
  } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<SP, MetalHalf>::value) {
    auto nvalue = (MetalHalf *)malloc(sizeof(MetalHalf) * rcount);
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              nvalue[jx] = (src[ix]);
            }
          }
        }
      }
    } else if (tensor_dim_.size() == 3) {
        for (int i0 = 0; i0 < N; ++i0) {
            for (int i1 = 0; i1 < C; ++i1) {
                for (int i2 = 0; i2 < H; ++i2) {
                    for (int i3 = 0; i3 < W; ++i3) {
                        std::vector<int> ig = {i0, i1, i2, i3};
                        auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
                        std::vector<int> jg = {
                                ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
                        auto k = jg[1];
                        auto jx =
                                ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
                        nvalue[jx] = src[ix];
                    }
                }
            }
        }

    } else {
        for (int i0 = 0; i0 < C; ++i0) {
            for (int i1 = 0; i1 < H; ++i1) {
                for (int i2 = 0; i2 < W; ++i2) {
                    auto ix = (i0 * W * H) + (i1 * W) + i2;
                    auto jx = ((i0 / 4) * texture_width_ * 4 * H) + (i1 * texture_width_ * 4) + i2;
                    nvalue[jx] = src[ix];
                }
            }
        }
    }

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
    auto bytes_per_image = image_.height * bytes_per_row;
    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};

    for (int i = 0; i < array_length_; ++i) {
      auto p = nvalue + image_.width * image_.height * channels_per_pixel_ * i;
      [image_ replaceRegion:region
                mipmapLevel:0
                      slice:static_cast<NSUInteger>(i)
                  withBytes:p
                bytesPerRow:bytes_per_row
              bytesPerImage:bytes_per_image];
    }

    free(nvalue);
  } else {
    throw std::logic_error("ERROR: do not support this half format");
    return;
  }
}

__unused void MetalImage::Zero() const {
  if (image_ == nullptr) return;

  int size_p = 1;
  if (precision_type_ == METAL_PRECISION_TYPE::FLOAT)
    size_p = 4;
  else if (precision_type_ == METAL_PRECISION_TYPE::HALF)
    size_p = 2;
  auto rcount = texture_width_ * texture_height_ * 1 * channels_per_pixel_;
  char *nvalue = (char *)malloc(size_p * rcount);
  memset(nvalue, 0, size_p * rcount);

  auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * size_p;
  auto bytes_per_image = image_.height * bytes_per_row;
  const MTLRegion region{.origin = {0, 0, 0},
                         .size = {
                             image_.width,
                             image_.height,
                             image_.depth,
                         }};

  {
    int i = static_cast<int>(array_length_ - 1);
    auto p = nvalue;
    [image_ replaceRegion:region
              mipmapLevel:0
                    slice:static_cast<NSUInteger>(i)
                withBytes:p
              bytesPerRow:bytes_per_row
            bytesPerImage:bytes_per_image];
  }

  free(nvalue);
}

template <typename P>
void MetalImage::CopyToNCHW(P *dst) const {
  size_t new_dims[] = {1, 1, 1, 1};
  for (int i = 0; i < tensor_dim_.size(); ++i) {
    new_dims[4 - tensor_dim_.size() + i] = static_cast<size_t>(tensor_dim_[i]);
  }

  size_t N = new_dims[0];
  size_t C = new_dims[1];
  size_t H = new_dims[2];
  size_t W = new_dims[3];

  auto dstCounts = array_length_ * channels_per_pixel_ * texture_width_ * texture_height_;

  if (precision_type_ == METAL_PRECISION_TYPE::FLOAT && std::is_same<P, float>::value) {
    auto pointer = (float *)malloc(sizeof(float) * dstCounts);

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(float);
    auto bytes_per_image = image_.height * bytes_per_row;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};
    for (int i = 0; i < array_length_; ++i) {
      auto p = pointer + image_.width * image_.height * channels_per_pixel_ * i;

      [image_ getBytes:(p)
           bytesPerRow:(bytes_per_row)bytesPerImage:(bytes_per_image)fromRegion:(region)mipmapLevel
                      :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              dst[ix] = pointer[jx];
            }
          }
        }
      }
    } else if (tensor_dim_.size() == 3) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[1];
              auto jx =
                  ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
              dst[ix] = pointer[jx];
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = pointer[texture_width_ * 4 * h + w];
        }
      }
    }
    free(pointer);
  } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, float>::value) {
    auto pointer = (MetalHalf *)malloc(sizeof(MetalHalf) * dstCounts);

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
    auto bytes_per_image = image_.height * bytes_per_row;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};
    for (int i = 0; i < array_length_; ++i) {
      auto p = pointer + image_.width * image_.height * channels_per_pixel_ * i;

      [image_ getBytes:(p)
           bytesPerRow:(bytes_per_row)bytesPerImage:(bytes_per_image)fromRegion:(region)mipmapLevel
                      :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              dst[ix] = MetalHalf2Float(pointer[jx]);
            }
          }
        }
      }

    } else if (tensor_dim_.size() == 3) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {ig };
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[1];
              auto jx =
                  ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
              dst[ix] = MetalHalf2Float(pointer[jx]);
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = MetalHalf2Float(pointer[texture_width_ * 4 * h + w]);
        }
      }
    }
    free(pointer);
  } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, MetalHalf>::value) {
    auto pointer = (MetalHalf *)malloc(sizeof(MetalHalf) * dstCounts);

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
    auto bytes_per_image = image_.height * bytes_per_row;

    const MTLRegion region{.origin = {0, 0, 0},
                           .size = {
                               image_.width,
                               image_.height,
                               image_.depth,
                           }};
    for (int i = 0; i < array_length_; ++i) {
      auto p = pointer + image_.width * image_.height * channels_per_pixel_ * i;

      [image_ getBytes:(p)
           bytesPerRow:(bytes_per_row)bytesPerImage:(bytes_per_image)fromRegion:(region)mipmapLevel
                      :(0)slice:static_cast<NSUInteger>(i)];
    }

    int index = 0;
    if (tensor_dim_.size() == 4) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[0] * dim_[3] + jg[3];
              auto jx =
                  ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) + (jg[2] * 4) + (k % 4);
              dst[ix] = pointer[jx];
            }
          }
        }
      }
    } else if (tensor_dim_.size() == 3) {
      for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < C; ++i1) {
          for (int i2 = 0; i2 < H; ++i2) {
            for (int i3 = 0; i3 < W; ++i3) {
              std::vector<int> ig = {i0, i1, i2, i3};
              auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
              std::vector<int> jg = {
                  ig[transpose_[0]], ig[transpose_[1]], ig[transpose_[2]], ig[transpose_[3]]};
              auto k = jg[1];
              auto jx =
                  ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) + (jg[3] * 4) + (k % 4);
              dst[ix] = MetalHalf2Float(pointer[jx]);
            }
          }
        }
      }
    } else {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          dst[index++] = pointer[texture_width_ * 4 * h + w];
        }
      }
    }
    free(pointer);
  }
}

MetalImage::~MetalImage() {
  if ( nil != image_){
#if (!__has_feature(objc_arc))
    [image_ release];
#endif
    image_ = nil;
  }

  device_ = nullptr;
}

template void MetalImage::CopyFromNCHW(const float *src);
template void MetalImage::CopyFromNCHW(const MetalHalf *src);
template void MetalImage::CopyToNCHW(float *dst) const;
template void MetalImage::CopyToNCHW(MetalHalf *dst) const;
}
}