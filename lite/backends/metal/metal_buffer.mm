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

#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_context_imp.h"
#include <cassert>

namespace paddle {
namespace lite {

MetalBuffer::MetalBuffer(MetalContext* context, size_t size, void* data, METAL_ACCESS_FLAG access)
    : mtl_size_(size) {
    can_copy_to_ = false;
    auto backend = (__bridge MetalContextImp*)context->backend();
    if (data) {
        buffer_ = [backend newDeviceBuffer:size bytes:data access:access];
    } else {
        buffer_ = [backend newDeviceBuffer:size access:access];
    }
}

MetalBuffer::MetalBuffer(MetalContext* context,
                         const DDim& inDim,
                         size_t size,
                         void* data,
                         METAL_PRECISION_TYPE precision,
                         METAL_ACCESS_FLAG access)
    : metal_context_(context),
      mtl_size_(size),
      dim_(inDim),
      tensor_dim_(inDim),
      precision_(precision),
      access_(access) {
    assert(precision_ == METAL_PRECISION_TYPE::FLOAT || precision_ == METAL_PRECISION_TYPE::HALF);

    if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
        precision_size_ = 4;
    } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
        precision_size_ = 2;
    }

    auto backend = (__bridge MetalContextImp*)context->backend();
    if (data) {
        buffer_ = [backend newDeviceBuffer:size bytes:data access:access];
    } else {
        buffer_ = [backend newDeviceBuffer:size access:access];
    }
}

MetalBuffer::MetalBuffer(MetalContext* context, const DDim& inDim, METAL_PRECISION_TYPE precision)
    : metal_context_(context), dim_(inDim), tensor_dim_(inDim), precision_(precision) {
    assert(precision_ == METAL_PRECISION_TYPE::FLOAT || precision_ == METAL_PRECISION_TYPE::HALF);

    if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
        precision_size_ = 4;
    } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
        precision_size_ = 2;
    }

    auto data_length = static_cast<size_t>(tensor_dim_.production() * precision_size_);
    rawdata_ = malloc(data_length);
}

template <>
void MetalBuffer::CopyFromNCHW(const float* src) {
    assert(src != nullptr);
    auto tensor_count = static_cast<int>(tensor_dim_.production());
    auto tensor_length = static_cast<size_t>(tensor_count * precision_size_);

    // rawdata赋值
    if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        memcpy(rawdata_, src, tensor_length);
    else if (precision_ == METAL_PRECISION_TYPE::HALF) {
        MetalFloatArray2HalfArray(src, (MetalHalf*)rawdata_, tensor_count);
    }

    //数据格式装换 NCHW -> NHWC
    if (convert_to_nhwc_) {
        Convert2NHWC();
        data_layout_ = DataLayout::kNHWC;
    }

    //反卷积等场景使用
    if (with_transpose_ && tensor_dim_.size() == 4) {
        auto transpose_pointer = malloc(tensor_length);
        auto n = tensor_dim_[0];
        auto hwc = tensor_dim_.production() / n;
        //
        if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
            auto data_src = (float*)rawdata_;
            auto data_dst = (float*)transpose_pointer;
            for (int j = 0; j < hwc; j++) {
                for (int i = 0; i < n; i++) {
                    data_dst[j * n + i] = data_src[i * hwc + j];
                }
            }
        } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
            auto data_src = (MetalHalf*)rawdata_;
            auto data_dst = (MetalHalf*)transpose_pointer;
            for (int j = 0; j < hwc; j++) {
                for (int i = 0; i < n; i++) {
                    data_dst[j * n + i] = data_src[i * hwc + j];
                }
            }
        }
        // swap the dim
        auto temp = dim_[0];
        dim_[0] = dim_[3];
        dim_[3] = temp;

        if (rawdata_ != nullptr) {
            free(rawdata_);
        }
        rawdata_ = transpose_pointer;
    }

    // rawdata(NHWC) 与 MTLBuffer 绑定
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    if (dim_.size() == 4) {
        int C = 0;
        if (data_layout_ == DataLayout::kNCHW) {
            C = static_cast<int>(dim_[1]);
        } else if (data_layout_ == DataLayout::kNHWC) {
            C = static_cast<int>(dim_[3]);
        } else {
            throw std::logic_error("ERROR: can only support NHWC and NCHW");
        }
        int cSlices = (C + 3) / 4;
        int paddedC = cSlices * 4;

        //不填充情况
        if (C == paddedC || (C == 1 && !pad_when_one_channel_)) {
            buffer_ = [backend newDeviceBuffer:(NSUInteger)tensor_length
                                        access:METAL_ACCESS_FLAG::CPUReadWrite];
            memcpy(buffer_.contents, rawdata_, tensor_length);
            mtl_size_ = tensor_length;
        }
        //填充
        else {
            if (data_layout_ == DataLayout::kNCHW) {
                //注意补充N通道 否则shader中gid.z取到边界时会报错（因为filter的weight数组是一维的）
                int padNCount =
                    static_cast<int>((((dim_[0] + 3) / 4) * 4 * paddedC) * dim_[2] * dim_[3]);
                void* convertedPointer = malloc(padNCount * precision_size_);
                //填充数据
                if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
                    auto data_src = (float*)rawdata_;
                    auto data_dst = (float*)convertedPointer;
                    int index1 = 0;
                    int index2 = 0;
                    for (int i = 0; i < dim_[0]; i++) {
                        for (int j = 0; j < paddedC; j++) {
                            for (int k = 0; k < dim_[2] * dim_[3]; k++) {
                                if (j < C) {
                                    data_dst[index2] = data_src[index1];
                                    index1 += 1;
                                    index2 += 1;
                                } else {
                                    data_dst[index2] = 0.0f;
                                    ;
                                    index2 += 1;
                                }
                            }
                        }
                    }
                } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
                    auto data_src = (MetalHalf*)rawdata_;
                    auto data_dst = (MetalHalf*)convertedPointer;
                    int index1 = 0;
                    int index2 = 0;
                    for (int i = 0; i < dim_[0]; i++) {
                        for (int j = 0; j < paddedC; j++) {
                            for (int k = 0; k < dim_[2] * dim_[3]; k++) {
                                if (j < C) {
                                    data_dst[index2] = data_src[index1];
                                    index1 += 1;
                                    index2 += 1;
                                } else {
                                    data_dst[index2] = 0.0f;
                                    ;
                                    index2 += 1;
                                }
                            }
                        }
                    }
                }
                //上传到GPU
                NSInteger new_length = padNCount * precision_size_;
                buffer_ = [backend newDeviceBuffer:new_length
                                            access:METAL_ACCESS_FLAG::CPUReadWrite];
                memcpy(buffer_.contents, convertedPointer, new_length);
                mtl_size_ = new_length;
                //释放填充数据
                if (convertedPointer != nullptr) {
                    free(convertedPointer);
                }
            } else if (data_layout_ == DataLayout::kNHWC) {
                int padNCount =
                    static_cast<int>((((dim_[0] + 3) / 4) * 4 * paddedC) * dim_[1] * dim_[2]);
                void* convertedPointer = malloc(padNCount * precision_size_);
                //填充数据
                if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
                    auto data_src = (float*)rawdata_;
                    auto data_dst = (float*)convertedPointer;
                    for (int i = 0; i < dim_[0] * dim_[1] * dim_[2]; i++) {
                        for (int j = 0; j < paddedC; j++) {
                            if (j < C) {
                                data_dst[j] = data_src[j];
                            } else {
                                data_dst[j] = 0.0f;
                            }
                        }
                        data_src += C;
                        data_dst += paddedC;
                    }
                } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
                    auto data_src = (MetalHalf*)rawdata_;
                    auto data_dst = (MetalHalf*)convertedPointer;
                    for (int i = 0; i < dim_[0] * dim_[1] * dim_[2]; i++) {
                        for (int j = 0; j < paddedC; j++) {
                            if (j < C) {
                                data_dst[j] = data_src[j];
                            } else {
                                data_dst[j] = 0.0f;
                            }
                        }
                        data_src += C;
                        data_dst += paddedC;
                    }
                }
                //上传到GPU
                NSInteger new_length = padNCount * precision_size_;
                buffer_ = [backend newDeviceBuffer:new_length
                                            access:METAL_ACCESS_FLAG::CPUReadWrite];
                memcpy(buffer_.contents, convertedPointer, new_length);
                mtl_size_ = new_length;
                //释放填充数据
                if (convertedPointer != nullptr) {
                    free(convertedPointer);
                }
            }
        }
    } else if (dim_.size() == 1) {
        buffer_ = [backend newDeviceBuffer:(NSUInteger)tensor_length
                                    access:METAL_ACCESS_FLAG::CPUReadWrite];
        memcpy(buffer_.contents, rawdata_, tensor_length);
        mtl_size_ = tensor_length;
    } else {
        throw std::logic_error("ERROR: can only support dim 1 and dim 4");
    }
}

template <>
void MetalBuffer::CopyToNCHW(float* dst) {
    assert(dst != nullptr);
    if (!can_copy_to_) {
        return;
    }

    void* src_ptr = nullptr;
    long size = 0;
    if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
        auto count = [buffer_ length] / sizeof(float);
        size = count * sizeof(float);
        auto float_ptr = malloc([buffer_ length]);
        memcpy(float_ptr, buffer_.contents, size);
        src_ptr = float_ptr;
    } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
        auto count = [buffer_ length] / sizeof(MetalHalf);
        size = count * sizeof(float);
        auto float_ptr = malloc(size);
        MetalHalfArray2FloatArray((MetalHalf*)buffer_.contents, (float*)float_ptr, (int)count);
        src_ptr = float_ptr;
    }

    if (data_layout_ == DataLayout::kNCHW) {
        memcpy((void*)dst, src_ptr, size);
    } else if (data_layout_ == DataLayout::kNHWC) {
        MetalConverter::NHWC2NCHW<float, float>(dst, (float*)src_ptr,
                                                static_cast<int>(tensor_dim_[0]),
                                                static_cast<int>(tensor_dim_[1]),
                                                static_cast<int>(tensor_dim_[2]),
                                                static_cast<int>(tensor_dim_[3]));
    }
    free(src_ptr);
}

template <typename P>
P* MetalBuffer::Convert(DataConverter<P>* converter) {
    auto cap = converter->Capacity(dim_);
    auto toCapacity = cap ? cap : dim_.production();
    auto to = (P*)malloc(sizeof(P) * toCapacity);
    try {
        converter->Convert(rawdata_, to, dim_);
    } catch (std::exception& error) {
        free(to);
        throw error;
    }
    free(rawdata_);
    rawdata_ = to;

    dim_ = converter->GetToDim(dim_);
    return to;
}

#pragma mark - internal

void MetalBuffer::Convert2NHWC() {
    if (tensor_dim_.size() != 4) return;
    void* new_pointer = malloc(static_cast<size_t>(precision_size_ * tensor_dim_.production()));

    if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
        MetalConverter::NCHW2NHWC<float, float>((float*)new_pointer, (float*)rawdata_,
                                                static_cast<int>(tensor_dim_[0]),
                                                static_cast<int>(tensor_dim_[1]),
                                                static_cast<int>(tensor_dim_[2]),
                                                static_cast<int>(tensor_dim_[3]));
    } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
        MetalConverter::NCHW2NHWC<MetalHalf, MetalHalf>((MetalHalf*)new_pointer,
                                                        (MetalHalf*)rawdata_,
                                                        static_cast<int>(tensor_dim_[0]),
                                                        static_cast<int>(tensor_dim_[1]),
                                                        static_cast<int>(tensor_dim_[2]),
                                                        static_cast<int>(tensor_dim_[3]));
    }

    int temp = static_cast<int>(dim_[3]);
    dim_[3] = dim_[1];
    dim_[1] = dim_[2];
    dim_[2] = temp;

    if (rawdata_ != nullptr) {
        free(rawdata_);
        rawdata_ = nullptr;
    }
    rawdata_ = new_pointer;
}

id<MTLBuffer> MetalBuffer::buffer() const {
    return buffer_;
}

MetalBuffer::~MetalBuffer() {
    if (buffer_ != nil) {
        buffer_ = nil;
    }
    if (rawdata_) {
        free(rawdata_);
        rawdata_ = nullptr;
    }
}

#pragma mark - useless

void MetalBuffer::Read(void* data, size_t size, size_t offset) const {
    if (buffer_ == nil) return;

    memcpy(data, (uint8_t*)[buffer_ contents] + offset, size);
}

void MetalBuffer::Write(const void* src, size_t size, size_t offset) const {
    if (buffer_ == nil) return;

    memcpy((uint8_t*)[buffer_ contents] + offset, src, size);
}

void MetalBuffer::Copy(const MetalBuffer& src,
                       size_t size,
                       size_t src_offset,
                       size_t dst_offset) const {
    if (buffer_ == nil) return;
    const auto& src_mtl_buffer = src;

    memcpy((uint8_t*)[buffer_ contents] + dst_offset,
           (uint8_t*)[src_mtl_buffer.buffer() contents] + src_offset, size);
}


}  // namespace lite
}  // namespace paddle
