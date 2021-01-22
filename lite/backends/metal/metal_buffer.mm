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
#include "lite/backends/metal/metal_queue.h"

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

metal_buffer::metal_buffer(const metal_device &device, size_t size, const METAL_ACCESS_FLAG flag)
    : flags_(flag) {
  mtl_device_ = const_cast<metal_device *>(&device);
  mtl_buffer_ = [device.get_device() newBufferWithLength:size options:option_for_access(flag)];
}

metal_buffer::metal_buffer(const metal_device &device,
                           void *data,
                           size_t size,
                           const METAL_ACCESS_FLAG flag)
    : flags_(flag) {
  mtl_device_ = const_cast<metal_device *>(&device);
  mtl_buffer_ = [device.get_device() newBufferWithBytes:data
                                                 length:size
                                                options:option_for_access(flag)];
}

metal_buffer::metal_buffer(const metal_device &device,
                           DDim inDim,
                           METAL_PRECISION_TYPE precision,
                           bool padWhenOneC,
                           bool convertToNHWC,
                           bool withTranspose,
                           METAL_ACCESS_FLAG flag)
    : tensor_dim_(inDim),
      precision_(precision),
      pad_when_one_channel_(padWhenOneC),
      convert_to_nhwc_(convertToNHWC),
      with_transpose_(withTranspose),
      dim_(inDim),
      flags_(flag) {
  assert(precision_ == METAL_PRECISION_TYPE::FLOAT || precision_ == METAL_PRECISION_TYPE::HALF);

  if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
    precision_size_ = 4;
  } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    precision_size_ = 2;
  }

  data_length_ = static_cast<size_t>(precision_size_ * inDim.production());
  data_ = malloc(data_length_);

  if (tensor_dim_.size() == 4) {
    c_ = static_cast<int>(tensor_dim_[1]);
    c_slices_ = (c_ + 3) / 4;
    padded_c_ = c_slices_ * 4;
    count_ = static_cast<int>(padded_c_ * tensor_dim_[0] * tensor_dim_[3] * tensor_dim_[2]);
    mtl_buffer_ =
        [device.get_device() newBufferWithLength:static_cast<NSUInteger>(count_ * precision_size_)
                                         options:option_for_access(flag)];
  } else if (tensor_dim_.size() == 1) {
    count_ = static_cast<int>(((tensor_dim_.production() + 3) / 4) * 4);
    mtl_buffer_ =
        [device.get_device() newBufferWithLength:static_cast<NSUInteger>(count_ * precision_size_)
                                         options:option_for_access(flag)];
  }
}

template <>
void metal_buffer::from_nchw(const float *src) {
  assert(src != nullptr);

  if (precision_ == METAL_PRECISION_TYPE::FLOAT)
    memcpy(data_, src, tensor_dim_.production() * precision_size_);
  else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    MetalFloatArray2HalfArray(src, (metal_half *)data_, static_cast<int>(tensor_dim_.production()));
  }
  if (convert_to_nhwc_) {
    convert();
  }

  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transposePointer = (void *)malloc(precision_size_ * tensor_dim_.production());
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transposePointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (metal_half *)data_;
        auto data_dst = (metal_half *)transposePointer;
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
    data_ = transposePointer;
    data_length_ = precision_size_ * dim_.production();
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      memcpy(mtl_buffer_.contents, data_, data_length_);
    } else {
      if (precision_ == METAL_PRECISION_TYPE::FLOAT) expand_nhwc<float>();
      if (precision_ == METAL_PRECISION_TYPE::HALF) expand_nhwc<metal_half>();
    }
  } else if (tensor_dim_.size() == 1) {
    memcpy(mtl_buffer_.contents, data_, data_length_);
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void metal_buffer::from_nchw(const metal_half *src) {
  assert(src != nullptr);
  if (precision_ == METAL_PRECISION_TYPE::FLOAT)
    MetalHalfArray2FloatArray(src, (float *)data_, static_cast<int>(tensor_dim_.production()));
  else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    memcpy(data_, src, tensor_dim_.production() * precision_size_);
  }
  if (convert_to_nhwc_) {
    convert();
  }

  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transposePointer = (void *)malloc(precision_size_ * tensor_dim_.production());
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transposePointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[j * n + i] = data_src[i * hwc + j];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (metal_half *)data_;
        auto data_dst = (metal_half *)transposePointer;
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
    data_ = transposePointer;
    data_length_ = precision_size_ * dim_.production();
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      memcpy(mtl_buffer_.contents, data_, data_length_);
    } else {
      if (precision_ == METAL_PRECISION_TYPE::FLOAT) expand_nhwc<float>();
      if (precision_ == METAL_PRECISION_TYPE::HALF) expand_nhwc<metal_half>();
    }
  } else if (tensor_dim_.size() == 1) {
    memcpy(mtl_buffer_.contents, data_, data_length_);
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void metal_buffer::to_nchw(metal_half *dst) {
  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transposePointer = (void *)malloc(precision_size_ * tensor_dim_.production());
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transposePointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (metal_half *)data_;
        auto data_dst = (metal_half *)transposePointer;
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
    data_ = transposePointer;
    data_length_ = precision_size_ * dim_.production();
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        metal_converter::NHWC2NCHW<metal_half, metal_half>(dst,
                                                           (metal_half *)data_,
                                                           static_cast<int>(tensor_dim_[0]),
                                                           static_cast<int>(tensor_dim_[1]),
                                                           static_cast<int>(tensor_dim_[2]),
                                                           static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        metal_converter::NHWC2NCHW<metal_half, float>(dst,
                                                      (float *)data_,
                                                      static_cast<int>(tensor_dim_[0]),
                                                      static_cast<int>(tensor_dim_[1]),
                                                      static_cast<int>(tensor_dim_[2]),
                                                      static_cast<int>(tensor_dim_[3]));
    } else {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        metal_converter::NHWC_EXPAND2NCHW<metal_half, metal_half>(dst,
                                                                  (metal_half *)data_,
                                                                  static_cast<int>(tensor_dim_[0]),
                                                                  static_cast<int>(tensor_dim_[1]),
                                                                  static_cast<int>(tensor_dim_[2]),
                                                                  static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        metal_converter::NHWC_EXPAND2NCHW<metal_half, float>(dst,
                                                             (float *)data_,
                                                             static_cast<int>(tensor_dim_[0]),
                                                             static_cast<int>(tensor_dim_[1]),
                                                             static_cast<int>(tensor_dim_[2]),
                                                             static_cast<int>(tensor_dim_[3]));
    }
  } else if (tensor_dim_.size() == 1) {
    if (precision_ == METAL_PRECISION_TYPE::FLOAT)
      MetalFloatArray2HalfArray((float *)data_,
                                (metal_half *)mtl_buffer_.contents,
                                static_cast<int>(tensor_dim_.production()));
    else if (precision_ == METAL_PRECISION_TYPE::HALF)
      memcpy((void *)dst, mtl_buffer_.contents, tensor_dim_.production() * sizeof(metal_half));
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <>
void metal_buffer::to_nchw(float *dst) {
  if (with_transpose_ && tensor_dim_.size() == 4) {
    auto transposePointer = (void *)malloc(precision_size_ * tensor_dim_.production());
    auto n = tensor_dim_[0];
    auto hwc = tensor_dim_.production() / n;

    switch (precision_) {
      case METAL_PRECISION_TYPE::FLOAT: {
        auto data_src = (float *)data_;
        auto data_dst = (float *)transposePointer;
        for (int j = 0; j < hwc; j++) {
          for (int i = 0; i < n; i++) {
            data_dst[i * hwc + j] = data_src[j * n + i];
          }
        }
        break;
      }
      case METAL_PRECISION_TYPE::HALF: {
        auto data_src = (metal_half *)data_;
        auto data_dst = (metal_half *)transposePointer;
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
    data_ = transposePointer;
    data_length_ = precision_size_ * dim_.production();
  }

  if (tensor_dim_.size() == 4) {
    if (c_ == padded_c_ || (c_ == 1 && !pad_when_one_channel_)) {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        metal_converter::NHWC2NCHW<float, metal_half>((float *)dst,
                                                      (metal_half *)data_,
                                                      static_cast<int>(tensor_dim_[0]),
                                                      static_cast<int>(tensor_dim_[1]),
                                                      static_cast<int>(tensor_dim_[2]),
                                                      static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        metal_converter::NHWC2NCHW<float, float>((float *)dst,
                                                 (float *)data_,
                                                 static_cast<int>(tensor_dim_[0]),
                                                 static_cast<int>(tensor_dim_[1]),
                                                 static_cast<int>(tensor_dim_[2]),
                                                 static_cast<int>(tensor_dim_[3]));
    } else {
      if (precision_ == METAL_PRECISION_TYPE::HALF)
        metal_converter::NHWC_EXPAND2NCHW<float, metal_half>((float *)dst,
                                                             (metal_half *)data_,
                                                             static_cast<int>(tensor_dim_[0]),
                                                             static_cast<int>(tensor_dim_[1]),
                                                             static_cast<int>(tensor_dim_[2]),
                                                             static_cast<int>(tensor_dim_[3]));
      else if (precision_ == METAL_PRECISION_TYPE::FLOAT)
        metal_converter::NHWC_EXPAND2NCHW<float, float>((float *)dst,
                                                        (float *)data_,
                                                        static_cast<int>(tensor_dim_[0]),
                                                        static_cast<int>(tensor_dim_[1]),
                                                        static_cast<int>(tensor_dim_[2]),
                                                        static_cast<int>(tensor_dim_[3]));
    }
  } else if (tensor_dim_.size() == 1) {
    if (precision_ == METAL_PRECISION_TYPE::FLOAT)
      memcpy((void *)dst, mtl_buffer_.contents, tensor_dim_.production() * sizeof(float));
    else if (precision_ == METAL_PRECISION_TYPE::HALF)
      MetalHalfArray2FloatArray((metal_half *)data_,
                                (float *)mtl_buffer_.contents,
                                static_cast<int>(tensor_dim_.production()));
  } else {
    throw std::logic_error("ERROR: can only support dim 1 and dim 4");
  }
}

template <typename P>
void metal_buffer::expand_nhwc() {
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
  memcpy(mtl_buffer_.contents, convertedPointer, static_cast<size_t>(data_length_));
  free(convertedPointer);
}

template <typename P>
P *metal_buffer::convert(data_converter<P> *converter) {
  auto cap = converter->capacity(dim_);
  auto toCapacity = cap ? cap : dim_.production();
  auto to = (P *)malloc(sizeof(P) * toCapacity);
  try {
    converter->convert(data_, to, dim_);
  } catch (std::exception &error) {
    free(to);
    throw error;
  }
  free(data_);
  data_ = to;
  data_length_ = static_cast<size_t>(toCapacity);

  dim_ = converter->getToDim(dim_);
  return to;
}

void metal_buffer::convert() {
  if (tensor_dim_.size() != 4) return;
  void *newPointer = malloc(static_cast<size_t>(precision_size_ * tensor_dim_.production()));

  if (precision_ == METAL_PRECISION_TYPE::FLOAT) {
    metal_converter::NCHW2NHWC<float, float>((float *)newPointer,
                                             (float *)data_,
                                             static_cast<int>(tensor_dim_[0]),
                                             static_cast<int>(tensor_dim_[1]),
                                             static_cast<int>(tensor_dim_[2]),
                                             static_cast<int>(tensor_dim_[3]));
  } else if (precision_ == METAL_PRECISION_TYPE::HALF) {
    metal_converter::NCHW2NHWC<metal_half, metal_half>((metal_half *)newPointer,
                                                       (metal_half *)data_,
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
  data_ = newPointer;
  data_length_ = static_cast<size_t>(precision_size_ * tensor_dim_.production());
}

id<MTLBuffer> metal_buffer::get_buffer() const { return mtl_buffer_; }

metal_buffer::~metal_buffer() {
  mtl_buffer_ = nil;
  if (data_) free(data_);
  data_ = nullptr;
}

void metal_buffer::read(void *data, const size_t size, const size_t offset) const {
  if (mtl_buffer_ == nil) return;

  const size_t actual_size = (size == 0 ? size_ : size);

  //    std::lock_guard<std::recursive_mutex> lock(buffer_lock_);
  memcpy(data, (uint8_t *)[mtl_buffer_ contents] + offset, actual_size);
}

void metal_buffer::write(const void *src,
                         const size_t size,
                         const size_t offset,
                         const metal_queue &queue) {
  write(src, size, offset);
}

void metal_buffer::write(const void *src, const size_t size, const size_t offset) const {
  if (mtl_buffer_ == nil) return;

  const size_t actual_size = (size == 0 ? size_ : size);
  memcpy((uint8_t *)[mtl_buffer_ contents] + offset, src, actual_size);
}

void metal_buffer::read(void *data,
                        const size_t size,
                        const size_t offset,
                        const metal_queue &queue) {
  read(data, size, offset);
}

void metal_buffer::copy(const metal_queue &queue,
                        const metal_buffer &src,
                        const size_t size,
                        const size_t src_offset,
                        const size_t dst_offset) {
  copy(src, size, src_offset, dst_offset);
}

metal_buffer::metal_buffer(const metal_device &device, void *mtl_buffer) {
  mtl_buffer_ = (__bridge id<MTLBuffer>)mtl_buffer;
  mtl_device_ = const_cast<metal_device *>(&device);
}

void metal_buffer::copy(const metal_buffer &src,
                        const size_t size,
                        const size_t src_offset,
                        const size_t dst_offset) const {
  if (mtl_buffer_ == nil) return;

  const auto &src_mtl_buffer = src;
  const size_t src_size = src.size_;
  const size_t copy_size = (size == 0 ? std::min(src_size, size_) : size);

  // std::lock_guard<std::recursive_mutex> lock(buffer_lock_);
  memcpy((uint8_t *)[mtl_buffer_ contents] + dst_offset,
         (uint8_t *)[src_mtl_buffer.get_buffer() contents] + src_offset,
         copy_size);
}

int metal_buffer::get_offset() const { return offset_; }
void metal_buffer::set_offset(int offset) { offset_ = offset; }

}
}