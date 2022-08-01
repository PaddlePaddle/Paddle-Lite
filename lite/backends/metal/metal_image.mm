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
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_half.h"
#include "lite/backends/metal/target_wrapper.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

MetalImage::MetalImage(MetalContext* context,
    const DDim& in_dim,
    std::vector<int> in_transpose,
    const METAL_PRECISION_TYPE precision_type,
    const METAL_ACCESS_FLAG flag,
    bool use_mps)
    : precision_type_(precision_type), flag_(flag), use_mps_(use_mps) {
    auto four_dim = FourDimFrom(in_dim);

    tensor_dim_ = in_dim;
    dim_ = four_dim;
    pad_to_four_dim_ = four_dim;
    transpose_ = std::move(in_transpose);

    InitTexture();
}

void MetalImage::InitTexture() {
    desc_ = [[MTLTextureDescriptor alloc] init];
    [desc_ setTextureType:MTLTextureType2DArray];
    [desc_ setDepth:1];

    std::vector<int> transpose_nhwc = {0, 2, 3, 1};
    std::vector<int> transpose_nchw = {0, 1, 2, 3};
    std::vector<DDim::value_type> dim = {};
    switch (tensor_dim_.size()) {
        case 4: {
            // attention: only support N=1
            // eg:tensor={1,24,208,208} to dim={1,208,208,24}
            // size of 'arraylength' direction is 24 ,value = (24+3)/4=6
            std::vector<int> transpose = {0, 2, 3, 1};
            for_each(transpose.begin(), transpose.end(), [&](int i) -> void {
                dim.emplace_back(pad_to_four_dim_[i]);
            });
        } break;
        case 3:
        case 2:
        case 1: {
            // attention: data arrangement ruler
            // eg1:tensor={1,36,3549} to dim={1,1,36,3549}
            // direction 'arraylength' use '3549' value=(3549+3)/4=888;
            // direction 'width' use '36' value=36  direction 'height' use '1' value=1
            // eg2:tensor={1,24} to dim={1,1,1,24}
            // direction 'arraylength' use '24' value=(24+3)/4=6;
            // direction 'width' use '1' value=1  direction 'height' use '1' value=1
            std::vector<int> transpose = {0, 1, 2, 3};
            for_each(transpose.begin(), transpose.end(), [&](int i) -> void {
                dim.emplace_back(pad_to_four_dim_[i]);
            });
        } break;
        default:
            LOG(FATAL) << "metal_image: Dim size is error";
    }
    dim_ = DDimLite(dim);

    //  data layout on GPU
    if (transpose_ == transpose_nhwc) {
        // scenes: Input of the previous kernel, output of the next kernel
        // attention: tensor.size=1、2、3、4
        switch (tensor_dim_.size()) {
            case 4:
                desc_.width = static_cast<NSUInteger>(dim[2]);
                desc_.height = static_cast<NSUInteger>(dim[1]);
                desc_.arrayLength = static_cast<NSUInteger>(((dim[0]) * (dim[3]) + 3) / 4);
                break;
            case 3:
                dim_ = DDimLite(dim);
                desc_.width = static_cast<NSUInteger>(dim[2]);
                desc_.height = static_cast<NSUInteger>(dim[1]);
                desc_.arrayLength = static_cast<NSUInteger>(((dim[0]) * (dim[3]) + 3) / 4);
                break;
            case 2:
            case 1:
                desc_.width = static_cast<NSUInteger>(dim[2]);
                desc_.height = static_cast<NSUInteger>(dim[1]);
                desc_.arrayLength = static_cast<NSUInteger>(((dim[3] + 3) / 4) * (dim[0]));
                break;
            default:
                LOG(FATAL) << "metal_image: Dim size is error";
        }
    } else if (transpose_ == transpose_nchw) {
        // scenes: the parameters used in the kernel calculation
        // eg: conv2d biasTexture, which come from io_copy_host_to_metal
        // attention: tensor.size=1、2
        switch (tensor_dim_.size()) {
            case 4:
            case 3:
            case 2:
            case 1: {
                desc_.width = static_cast<NSUInteger>(dim[2]);
                desc_.height = static_cast<NSUInteger>(dim[1]);
                desc_.arrayLength = static_cast<NSUInteger>(((dim[3] + 3) / 4) * (dim[0]));
            } break;
            default:
                LOG(FATAL) << "metal_image: Dim size is error";
        }
    } else {
        LOG(FATAL) << "metal_image: unsupported tensor dim count";
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
    desc_.storageMode = MTLStorageModeShared;
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
        LOG(FATAL) << "Error: cannot support such dims";
    }
    return four_dim;
}

id<MTLTexture> MetalImage::image() const {
    return image_;
}

void MetalImage::initImage(MetalContext* context) {
    auto backend = (__bridge MetalContextImp*)context->backend();
    image_ = [backend newTextureWithDescriptor:desc_];
}

void MetalImage::initImageReuse(MetalContext* context, std::string ptr) {
    if (@available(iOS 10.0, *)) {
        initImageFromHeap(context, ptr);
    } else {
        initImage(context);
    }
}

API_AVAILABLE(ios(10.0))
void MetalImage::initImageFromHeap(MetalContext* context, std::string ptr) {
    auto backend = (__bridge MetalContextImp*)context->backend();
    id<MTLHeap> heap = [backend getHeap:ptr];
    if (!heap) {
        heap = [backend newHeapWithDescriptor:desc_];
        [backend setHeap:heap key:ptr];
    }
    bool flag = [backend isNewHeapWithDescriptor:desc_ heap:heap];
    if (flag) {
        heap = [backend newHeapWithDescriptor:desc_];
        [backend setHeap:heap key:ptr];
    }
    heap_ = heap;
    image_ = [backend newTextureWithDescriptor:desc_ heap:heap];
}

int MetalImage::ElementCount() const {
    if (image_) {
        return (int)(image_.width * image_.height * image_.arrayLength * 4);
    } else if (desc_) {
        return (int)(desc_.width * desc_.height * desc_.arrayLength * 4);
    } else {
        LOG(FATAL) << "metal_image: texture desc = nil";
    }
}

// the same logic with above 'texture desc'
// scene1: the parameters used in the kernel calculation
// eg: io_copy_host_to_metal, tensor.size=1,2
// scene2: the unimplemented kernel in metal,
//         CPU process data and return to metal
// eg: io_copy_host_to_metal, tensor.size=4
template <typename SP>
void MetalImage::CopyFromNCHW(const SP* src) {
    auto rcount = texture_width_ * texture_height_ * array_length_ * channels_per_pixel_;

    if (precision_type_ == METAL_PRECISION_TYPE::FLOAT && std::is_same<SP, float>::value) {
        LOG(FATAL) << "metal_image: CopyFromNCHW - precision = FLOAT";
    } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<SP, float>::value) {
        // attenion：ruler same with InitTexture
        auto nvalue = (MetalHalf*)TargetWrapperMetal::Malloc(sizeof(MetalHalf) * rcount);
        TargetWrapperMetal::MemsetSync(nvalue, 0, sizeof(MetalHalf) * rcount);
        if (tensor_dim_.size() == 4) {
            size_t n = (size_t)dim_[0];
            size_t h = (size_t)dim_[1];
            size_t w = (size_t)dim_[2];
            size_t c = (size_t)dim_[3];
            size_t p = (c * n + 3) / 4;
            size_t C = n * c;
            for (int i1 = 0; i1 < p; i1++) {
                for (int i2 = 0; i2 < h; i2++) {
                    for (int i3 = 0; i3 < w; i3++) {
                        for (int k = 0; k < 4; k++) {
                            auto dx = i1 * h * w * 4 + i2 * w * 4 + i3 * 4 + k;
                            auto sx = (i1 * 4 + k) * h * w + i2 * w + i3;
                            if ((i1 * 4 + k) < C) {
                                nvalue[dx] = MetalFloat2Half(src[sx]);
                            } else {
                                nvalue[dx] = 0.0;
                            }
                        }
                    }
                }
            }
        } else if (tensor_dim_.size() == 3) {
            // LOG(FATAL) << "MetalImage: CopyFromNCHW - tensor dim = 3";
        } else {
            // dimension bellow 4 similar to texture desc
            size_t n = (size_t)dim_[0];
            size_t h = (size_t)dim_[1];
            size_t w = (size_t)dim_[2];
            size_t c = (size_t)dim_[3];
            size_t p = (c + 3) / 4;
            for (int i0 = 0; i0 < n; i0++) {
                for (int i1 = 0; i1 < p; i1++) {
                    for (int i2 = 0; i2 < h; i2++) {
                        for (int i3 = 0; i3 < w; i3++) {
                            for (int k = 0; k < 4; k++) {
                                auto dx =
                                    i0 * p * h * w * 4 + i1 * h * w * 4 + i2 * w * 4 + i3 * 4 + k;
                                auto sx = k + i1 * 4 + i3 * c + i2 * w * c + i0 * c * h * w;
                                if ((i1 * 4 + k) < c) {
                                    nvalue[dx] = MetalFloat2Half(src[sx]);
                                } else {
                                    nvalue[dx] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
        auto bytes_per_image = image_.height * bytes_per_row;
        const MTLRegion region{.origin = {0, 0, 0},
            .size =
                {
                    image_.width, image_.height, image_.depth,
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

        TargetWrapperMetal::Free(nvalue);
    } else {
        LOG(FATAL) << "ERROR: do not support this half format";
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
    char* nvalue = (char*)TargetWrapperMetal::Malloc(size_p * rcount);
    TargetWrapperMetal::MemsetSync(nvalue, 0, size_p * rcount);

    auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * size_p;
    auto bytes_per_image = image_.height * bytes_per_row;
    const MTLRegion region{.origin = {0, 0, 0},
        .size =
            {
                image_.width, image_.height, image_.depth,
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

    TargetWrapperMetal::Free(nvalue);
}

template <typename P>
void MetalImage::CopyToNCHW(P* dst) const {
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
        auto pointer = (float*)TargetWrapperMetal::Malloc(sizeof(float) * dstCounts);

        auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(float);
        auto bytes_per_image = image_.height * bytes_per_row;

        const MTLRegion region{.origin = {0, 0, 0},
            .size =
                {
                    image_.width, image_.height, image_.depth,
                }};
        for (int i = 0; i < array_length_; ++i) {
            auto p = pointer + image_.width * image_.height * channels_per_pixel_ * i;

            [image_ getBytes:(p)
                  bytesPerRow:(bytes_per_row)
                bytesPerImage:(bytes_per_image)
                   fromRegion:(region)
                  mipmapLevel:(0)
                        slice:static_cast<NSUInteger>(i)];
        }

        int index = 0;
        if (tensor_dim_.size() == 4) {
            for (int i0 = 0; i0 < N; ++i0) {
                for (int i1 = 0; i1 < C; ++i1) {
                    for (int i2 = 0; i2 < H; ++i2) {
                        for (int i3 = 0; i3 < W; ++i3) {
                            std::vector<int> ig = {i0, i1, i2, i3};
                            auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
                            std::vector<int> jg = {ig[transpose_[0]],
                                ig[transpose_[1]],
                                ig[transpose_[2]],
                                ig[transpose_[3]]};
                            auto k = jg[0] * dim_[3] + jg[3];
                            auto jx = ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) +
                                      (jg[2] * 4) + (k % 4);
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
                            std::vector<int> jg = {ig[transpose_[0]],
                                ig[transpose_[1]],
                                ig[transpose_[2]],
                                ig[transpose_[3]]};
                            auto k = jg[1];
                            auto jx = ((k / 4) * dim_[2] * dim_[3] * 4) + (jg[1] * dim_[3] * 4) +
                                      (jg[3] * 4) + (k % 4);
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
        TargetWrapperMetal::Free(pointer);
    } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, float>::value) {
        auto pointer = (MetalHalf*)TargetWrapperMetal::Malloc(sizeof(MetalHalf) * dstCounts);
        TargetWrapperMetal::MemsetSync(pointer, 0, sizeof(MetalHalf) * dstCounts);

        auto bytes_per_row = image_.width * image_.depth * channels_per_pixel_ * sizeof(MetalHalf);
        auto bytes_per_image = image_.height * bytes_per_row;

        const MTLRegion region{.origin = {0, 0, 0},
            .size =
                {
                    image_.width, image_.height, image_.depth,
                }};
        for (int i = 0; i < array_length_; ++i) {
            auto p = pointer + image_.width * image_.height * channels_per_pixel_ * i;

            [image_ getBytes:p
                  bytesPerRow:bytes_per_row
                bytesPerImage:bytes_per_image
                   fromRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(i)];
        }

        int index = 0;
        if (tensor_dim_.size() == 4) {
            for (int i0 = 0; i0 < N; ++i0) {
                for (int i1 = 0; i1 < C; ++i1) {
                    for (int i2 = 0; i2 < H; ++i2) {
                        for (int i3 = 0; i3 < W; ++i3) {
                            std::vector<int> ig = {i0, i1, i2, i3};
                            auto ix = (i0 * C * H * W) + (i1 * H * W) + (i2 * W) + i3;
                            std::vector<int> jg = {ig[transpose_[0]],
                                ig[transpose_[1]],
                                ig[transpose_[2]],
                                ig[transpose_[3]]};
                            auto k = jg[0] * dim_[3] + jg[3];
                            auto jx = ((k / 4) * dim_[1] * dim_[2] * 4) + (jg[1] * dim_[2] * 4) +
                                      (jg[2] * 4) + (k % 4);
                            dst[ix] = MetalHalf2Float(pointer[jx]);
                        }
                    }
                }
            }
        } else if (tensor_dim_.size() == 3) {
            size_t N = new_dims[0];
            size_t H = new_dims[1];
            size_t W = new_dims[2];
            size_t C = new_dims[3];

            for (int i0 = 0; i0 < N; ++i0) {
                for (int i1 = 0; i1 < H; ++i1) {
                    for (int i2 = 0; i2 < W; ++i2) {
                        for (int i3 = 0; i3 < C; ++i3) {
                            auto dx = (i0 * H * W * C) + (i1 * W * C) + (i2 * C) + i3;
                            auto sx = (i3 / 4) * H * W * 4 + i2 * 4 + i1 * W * 4 + i3 % 4;
                            dst[dx] = MetalHalf2Float(pointer[sx]);
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
        TargetWrapperMetal::Free(pointer);
    } else if (precision_type_ == METAL_PRECISION_TYPE::HALF && std::is_same<P, MetalHalf>::value) {
        LOG(FATAL) << "metal_image: CopyToNCHW-half2half";
    }
}

MetalImage::~MetalImage() {
    if (image_) {
        image_ = nil;
    }
    if (@available(iOS 10.0, *)) {
        if (heap_) {
            heap_ = nil;
        }
    }
}

template void MetalImage::CopyFromNCHW(const float* src);
template void MetalImage::CopyFromNCHW(const MetalHalf* src);
template void MetalImage::CopyToNCHW(float* dst) const;
template void MetalImage::CopyToNCHW(MetalHalf* dst) const;
}
}
