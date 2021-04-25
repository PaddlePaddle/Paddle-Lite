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

#include <cassert>
#include "lite/backends/metal/metal_buffer.h"

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

MetalBuffer::MetalBuffer(const MetalDevice &device, const MetalBufferDescriptor &desc) {
  MetalBuffer(device,
              desc.dim_,
              desc.precision_,
              desc.pad_when_one_c_,
              desc.convert_to_NHWC_,
              desc.with_transpose_,
              desc.flag_);
}

MetalBuffer::MetalBuffer(const MetalDevice &device, size_t size, const METAL_ACCESS_FLAG flag)
    : flags_(flag), type_(TYPE::kCommonBuffer), data_length_(size) {
  mtl_device_ = const_cast<MetalDevice *>(&device);
  buffer_ = [device.device() newBufferWithLength:size options:OptionForAccess(flag)];
}

MetalBuffer::MetalBuffer(const MetalDevice &device,
                         void *data,
                         size_t size,
                         const METAL_ACCESS_FLAG flag)
    : flags_(flag), type_(TYPE::kCommonBuffer), data_length_(size) {
  mtl_device_ = const_cast<MetalDevice *>(&device);
  buffer_ = [device.device() newBufferWithBytes:data length:size options:OptionForAccess(flag)];
}

MetalBuffer::MetalBuffer(const MetalDevice &device,
                         const DDim& in_dim,
                         METAL_PRECISION_TYPE precision,
                         bool pad_when_one_c,
                         bool convert_to_nhwc,
                         bool with_transpose,
                         METAL_ACCESS_FLAG flag)
    : tensor_dim_(in_dim),
      precision_(precision),
      pad_when_one_channel_(pad_when_one_c),
      convert_to_nhwc_(convert_to_nhwc),
      with_transpose_(with_transpose),
      dim_(in_dim),
      flags_(flag),
      type_(TYPE::kTensorBuffer) {
  assert(precision_ == METAL_PRECISION_TYPE::FLOAT ||
         precision_ == METAL_PRECISION_TYPE::HALF);

  if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
    precision_size_ = 4;
  } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    precision_size_ = 2;
  }

  data_length_ = static_cast<size_t>(precision_size_ * in_dim.production());
  data_ = malloc(data_length_);

  if (tensor_dim_.size() == 4) {
    c_ = static_cast<int>(tensor_dim_[1]);
    c_slices_ = (c_ + 3) / 4;
    padded_c_ = c_slices_ * 4;
    count_ = static_cast<int>(padded_c_ * tensor_dim_[0] * tensor_dim_[3] *
                              tensor_dim_[2]);
    buffer_ = [device.device()
        newBufferWithLength:static_cast<NSUInteger>(count_ * precision_size_)
                    options:OptionForAccess(flag)];
  } else if (tensor_dim_.size() == 1) {
    count_ = static_cast<int>(((tensor_dim_.production() + 3) / 4) * 4);
    buffer_ = [device.device()
        newBufferWithLength:static_cast<NSUInteger>(count_ * precision_size_)
                    options:OptionForAccess(flag)];
  }
}

template <>
void MetalBuffer::CopyFromNCHW<float>(const float *src) {
  assert(src != nullptr);
  auto tensor_length = static_cast<size_t>(tensor_dim_.production() * precision_size_);
  if (precision_ == METAL_PRECISION_TYPE::FLOAT)
    memcpy(data_, src, tensor_length);
  else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    MetalFloatArray2HalfArray(
        src, (MetalHalf *)data_, static_cast<int>(tensor_dim_.production()));
  }
  if (convert_to_nhwc_) {
    Convert();
  }

  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transpose_pointer = malloc(tensor_length);
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (MetalHalf *)data_;
        auto data_dst = (MetalHalf *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      default:
        throw std::logic_error("can only support compute half and float");
    }

    // swap the dim
    auto temp = dim_[0];
    dim_[0] = dim_[3];
    dim_[3] = temp;

    if (data_ != nullptr) free(data_);
    data_ = transpose_pointer;
    data_length_ = static_cast<size_t>(precision_size_ * dim_.production());
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      memcpy(buffer_.contents, data_, data_length_);
    } else {
      if (precision_ == METAL_PRECISION_TYPE::FLOAT) ExpandNHWC<float>();
      if (precision_ == METAL_PRECISION_TYPE::HALF) ExpandNHWC<MetalHalf>();
    }
  } else if (tensor_dim_.size() == 1) {
    memcpy(buffer_.contents, data_, data_length_);
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void MetalBuffer::CopyFromNCHW<MetalHalf>(const MetalHalf *src) {
  assert(src != nullptr);
  auto tensor_length = static_cast<size_t>(tensor_dim_.production() * precision_size_);
  if (precision_ == METAL_PRECISION_TYPE::FLOAT)
    MetalHalfArray2FloatArray(src, (float *)data_, static_cast<int>(tensor_dim_.production()));
  else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    memcpy(data_, src, tensor_length);
  }
  if (convert_to_nhwc_) {
    Convert();
  }

  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transpose_pointer = malloc(tensor_length);
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (MetalHalf *)data_;
        auto data_dst = (MetalHalf *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      default:
        throw std::logic_error("can only support compute half and float");
    }

    // swap the dim
    auto temp = dim_[0];
    dim_[0] = dim_[3];
    dim_[3] = temp;

    if (data_ != nullptr) free(data_);
    data_ = transpose_pointer;
    data_length_ = static_cast<size_t>(precision_size_ * dim_.production());
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      memcpy(buffer_.contents, data_, data_length_);
    } else {
      if (precision_ == METAL_PRECISION_TYPE::FLOAT) ExpandNHWC<float>();
      if (precision_ == METAL_PRECISION_TYPE::HALF) ExpandNHWC<MetalHalf>();
    }
  } else if (tensor_dim_.size() == 1) {
    memcpy(buffer_.contents, data_, data_length_);
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void MetalBuffer::CopyToNCHW<MetalHalf>(MetalHalf *dst) {
  assert(dst != nullptr);
  auto tensor_length = static_cast<size_t>(precision_size_ * tensor_dim_.production());
  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transpose_pointer = malloc(tensor_length);
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (MetalHalf *)data_;
        auto data_dst = (MetalHalf *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      default:
        throw std::logic_error("can only support compute half and float");
    }

    // swap the dim
    auto temp = dim_[0];
    dim_[0] = dim_[3];
    dim_[3] = temp;

    if (data_ != nullptr) free(data_);
    data_ = transpose_pointer;
    data_length_ = static_cast<size_t>(precision_size_ * dim_.production());
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        MetalConverter::NHWC2NCHW<MetalHalf, MetalHalf>(dst,
                                                        (MetalHalf *)data_,
                                                        static_cast<int>(tensor_dim_[0]),
                                                        static_cast<int>(tensor_dim_[1]),
                                                        static_cast<int>(tensor_dim_[2]),
                                                        static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        MetalConverter::NHWC2NCHW<MetalHalf, float>(dst,
                                                    (float *)data_,
                                                    static_cast<int>(tensor_dim_[0]),
                                                    static_cast<int>(tensor_dim_[1]),
                                                    static_cast<int>(tensor_dim_[2]),
                                                    static_cast<int>(tensor_dim_[3]));
    } else {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        MetalConverter::NHWCExpand2NCHW<MetalHalf, MetalHalf>(dst,
                                                              (MetalHalf *)data_,
                                                              static_cast<int>(tensor_dim_[0]),
                                                              static_cast<int>(tensor_dim_[1]),
                                                              static_cast<int>(tensor_dim_[2]),
                                                              static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        MetalConverter::NHWCExpand2NCHW<MetalHalf, float>(dst,
                                                          (float *)data_,
                                                          static_cast<int>(tensor_dim_[0]),
                                                          static_cast<int>(tensor_dim_[1]),
                                                          static_cast<int>(tensor_dim_[2]),
                                                          static_cast<int>(tensor_dim_[3]));
    }
  } else if (tensor_dim_.size() == 1) {
    if (precision_ == METAL_PRECISION_TYPE::FLOAT)
      MetalFloatArray2HalfArray((float *)data_,
                                (MetalHalf *)buffer_.contents,
                                static_cast<int>(tensor_dim_.production()));
    else if (precision_ == METAL_PRECISION_TYPE::HALF)
      memcpy((void *)dst, buffer_.contents, tensor_dim_.production() * sizeof(MetalHalf));
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void MetalBuffer::CopyToNCHW<float>(float *dst) {
  assert(dst != nullptr);
  auto tensor_length = static_cast<size_t>(precision_size_ * tensor_dim_.production());
  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transpose_pointer = malloc(tensor_length);
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (MetalHalf *)data_;
        auto data_dst = (MetalHalf *)transpose_pointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      default:
        throw std::logic_error("can only support compute half and float");
    }
    // swap the dim
    auto temp = dim_[0];
    dim_[0] = dim_[3];
    dim_[3] = temp;

    if (data_ != nullptr) free(data_);
    data_ = transpose_pointer;
    data_length_ = precision_size_ * dim_.production();
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        MetalConverter::NHWC2NCHW<float, MetalHalf>(dst,
                                                    (MetalHalf *)data_,
                                                    static_cast<int>(tensor_dim_[0]),
                                                    static_cast<int>(tensor_dim_[1]),
                                                    static_cast<int>(tensor_dim_[2]),
                                                    static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        MetalConverter::NHWC2NCHW<float, float>(dst,
                                                (float *)data_,
                                                static_cast<int>(tensor_dim_[0]),
                                                static_cast<int>(tensor_dim_[1]),
                                                static_cast<int>(tensor_dim_[2]),
                                                static_cast<int>(tensor_dim_[3]));
    } else {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        MetalConverter::NHWCExpand2NCHW<float, MetalHalf>(dst,
                                                          (MetalHalf *)data_,
                                                          static_cast<int>(tensor_dim_[0]),
                                                          static_cast<int>(tensor_dim_[1]),
                                                          static_cast<int>(tensor_dim_[2]),
                                                          static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        MetalConverter::NHWCExpand2NCHW<float, float>(dst,
                                                      (float *)data_,
                                                      static_cast<int>(tensor_dim_[0]),
                                                      static_cast<int>(tensor_dim_[1]),
                                                      static_cast<int>(tensor_dim_[2]),
                                                      static_cast<int>(tensor_dim_[3]));
    }
  } else if (tensor_dim_.size() == 1) {
    if (precision_ == METAL_PRECISION_TYPE::FLOAT)
      memcpy((void *)dst, buffer_.contents, tensor_dim_.production() * sizeof(float));
    else if (precision_ == METAL_PRECISION_TYPE::HALF)
      MetalHalfArray2FloatArray((MetalHalf *)data_,
                                (float *)buffer_.contents,
                                static_cast<int>(tensor_dim_.production()));
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <typename P>
void MetalBuffer::ExpandNHWC() {
  void *convertedPointer = malloc(count_ * sizeof(P));
  data_length_ = count_ * sizeof(P);
  P *tmpPointer = (P *)data_;
  P *dstPtr = (P *)convertedPointer;
  for (int i = 0; i < dim_[0] * dim_[1] * dim_[2]; i++) {
    for (int j = 0; j < padded_c_; j++) {
      if (j < c_) {
        dstPtr[j] = tmpPointer[j];
      } else {
        dstPtr[j] = 0.0f;
      }
    }
    tmpPointer += c_;
    dstPtr += padded_c_;
  }
  memcpy(buffer_.contents, convertedPointer, static_cast<size_t>(data_length_));
  free(convertedPointer);
}

template <typename P>
P *MetalBuffer::Convert(DataConverter<P> *converter) {
  auto cap = converter->Capacity(dim_);
  auto toCapacity = cap ? cap : dim_.production();
  auto to = (P *)malloc(sizeof(P) * toCapacity);
  try {
    converter->Convert(data_, to, dim_);
  } catch (std::exception &error) {
    free(to);
    throw error;
  }
  free(data_);
  data_ = to;
  data_length_ = static_cast<size_t>(toCapacity);

  dim_ = converter->GetToDim(dim_);
  return to;
}

void MetalBuffer::Convert() {
  if (tensor_dim_.size() != 4) return;
  void *new_pointer =
      malloc(static_cast<size_t>(precision_size_ * tensor_dim_.production()));

  if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
    MetalConverter::NCHW2NHWC<float, float>((float *)new_pointer,
                                            (float *)data_,
                                            static_cast<int>(tensor_dim_[0]),
                                            static_cast<int>(tensor_dim_[1]),
                                            static_cast<int>(tensor_dim_[2]),
                                            static_cast<int>(tensor_dim_[3]));
  } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    MetalConverter::NCHW2NHWC<MetalHalf, MetalHalf>(
        (MetalHalf *)new_pointer,
        (MetalHalf *)data_,
        static_cast<int>(tensor_dim_[0]),
        static_cast<int>(tensor_dim_[1]),
        static_cast<int>(tensor_dim_[2]),
        static_cast<int>(tensor_dim_[3]));
  }

  int temp = static_cast<int>(dim_[3]);
  dim_[3] = dim_[1];
  dim_[1] = dim_[2];
  dim_[2] = temp;

  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
  data_ = new_pointer;
  data_length_ =
      static_cast<size_t>(precision_size_ * tensor_dim_.production());
}

id<MTLBuffer> MetalBuffer::buffer() const { return buffer_; }

MetalBuffer::~MetalBuffer() {
  if(buffer_ != nil){
#if (!__has_feature(objc_arc)) 
    [buffer_ release];
#endif
    buffer_ = nil;
  }

  if (data_) {
    free(data_);
    data_ = nullptr;
  }
}

void MetalBuffer::Read(void *data, size_t size, size_t offset) const {
  if (buffer_ == nil) return;

  const size_t actual_size = (size == 0 ? size_ : size);
  memcpy(data, (uint8_t *)[buffer_ contents] + offset, actual_size);
}

void MetalBuffer::Write(const void *src,
                        size_t size,
                        size_t offset,
                        const MetalQueue &queue) const {
  Write(src, size, offset);
}

void MetalBuffer::Write(const void *src, size_t size, size_t offset) const {
  if (buffer_ == nil) return;

  const size_t actual_size = (size == 0 ? size_ : size);
  memcpy((uint8_t *)[buffer_ contents] + offset, src, actual_size);
}

void MetalBuffer::Read(void *data,
                       size_t size,
                       size_t offset,
                       const MetalQueue &queue) const {
  Read(data, size, offset);
}

void MetalBuffer::Copy(const MetalQueue &queue,
                       const MetalBuffer &src,
                       size_t size,
                       size_t src_offset,
                       size_t dst_offset) const {
  Copy(src, size, src_offset, dst_offset);
}

#ifdef __OBJC__
MetalBuffer::MetalBuffer(const MetalDevice &device, id<MTLBuffer> buffer) {
  buffer_ = buffer;
  mtl_device_ = const_cast<MetalDevice *>(&device);
  type_ = TYPE::kUnknownBuffer;
}
#else
MetalBuffer::MetalBuffer(const MetalDevice &device, void *mtl_buffer) {
  buffer_ = (__bridge id<MTLBuffer>)mtl_buffer;
  mtl_device_ = const_cast<MetalDevice *>(&device);
  type_ = TYPE::kUnknownBuffer;
}
#endif

void MetalBuffer::Copy(const MetalBuffer &src,
                       size_t size,
                       size_t src_offset,
                       size_t dst_offset) const {
  if (buffer_ == nil) return;
  assert(size < (size_ - dst_offset));
  const auto &src_mtl_buffer = src;
  const size_t src_size = src.size_;
  const size_t copy_size = (size == 0) ? std::min(src_size, size_): size_;

  memcpy((uint8_t *)[buffer_ contents] + dst_offset,
         (uint8_t *)[src_mtl_buffer.buffer() contents] + src_offset,
         copy_size);
}

int MetalBuffer::offset() const { return offset_; }
void MetalBuffer::set_offset(int offset) { offset_ = offset; }
} // namespace lite
} // namespace paddle
