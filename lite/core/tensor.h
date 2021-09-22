// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef LITE_WITH_FPGA
#include "lite/backends/fpga/lite_tensor.h"
#endif

#ifndef LITE_WITH_FPGA

#ifdef LITE_WITH_METAL
#include "lite/backends/metal/target_wrapper.h"
#endif  // LITE_WITH_METAL

#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "lite/core/dim.h"
#include "lite/core/memory.h"
#include "lite/utils/replace_stl/stream.h"

#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/xpu_buffer.h"
#endif

namespace paddle {
namespace lite {

class TensorLite;

using Tensor = lite::TensorLite;

using LoD = std::vector<std::vector<uint64_t>>;

// A light-weight tensor implementation.
class TensorLite {
 public:
  TensorLite() : buffer_(std::make_shared<Buffer>()) {}
  explicit TensorLite(std::shared_ptr<Buffer> buffer) : buffer_(buffer) {}

  template <typename DType, typename DimT, TargetType Target>
  void Assign(const DType *data, const DimT &dim) {
    Resize(dim);
    auto *dst = mutable_data<DType, void>(Target);
    CopySync<Target>(
        dst, data, dim.production() * sizeof(DType), IoDirection::HtoD);
  }

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  const R *data() const {
    return reinterpret_cast<const R *>(static_cast<char *>(buffer_->data()) +
                                       offset_);
  }

  void Resize(const DDimLite &ddim) { dims_ = ddim; }
  void Resize(const std::vector<int64_t> &x) { dims_.ConstructFrom(x); }

  const DDimLite &dims() const { return dims_; }
  int64_t numel() const { return dims_.production(); }

  const LoD &lod() const { return lod_; }
  LoD *mutable_lod() { return &lod_; }
  void set_lod(const LoD &lod) { lod_ = lod; }

  PrecisionType precision() const { return precision_; }
  void set_precision(PrecisionType precision) { precision_ = precision; }

  bool persistable() const { return persistable_; }
  void set_persistable(bool persistable) { persistable_ = persistable; }

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data() {
    precision_ = lite_api::PrecisionTypeTrait<T>::Type();
    memory_size_ = dims_.production() * sizeof(T);
    buffer_->ResetLazy(target_, memory_size_);
    return reinterpret_cast<R *>(static_cast<char *>(buffer_->data()) +
                                 offset_);
  }

#ifdef LITE_WITH_OPENCL
  template <typename T, typename R = T>
  R *mutable_data(const size_t img_w,
                  const size_t img_h,
                  void *host_ptr = nullptr) {
    target_ = TARGET(kOpenCL);
    buffer_->ResetLazyImage2D<T>(target_, img_w, img_h, host_ptr);
    return static_cast<cl::Image2D *>(buffer_->data());
  }
#endif

#ifdef LITE_WITH_METAL
  template <class T>
  struct IsImage {
    enum { value = std::is_same<T, MetalImage>::value };
  };

  template <class T>
  struct IsBuffer {
    enum { value = std::is_same<T, MetalBuffer>::value };
  };

  template <typename T, typename R>
  typename std::enable_if<IsImage<R>::value, R>::type *mutable_data(
      MetalContext *context,
      const DDim &dim,
      std::vector<int> transport = {0, 2, 3, 1},
      bool reuse = true) {
    dims_ = dim;
    target_ = TARGET(kMetal);
    long ptr_this = reinterpret_cast<long>(this);
    std::string ptr;
    std::stringstream stream;
    stream << ptr_this;
    stream >> ptr;
    buffer_->ResetLazyMetalImage<T>(context, dim, transport, reuse, ptr);
    return static_cast<MetalImage *>(buffer_->data());
  }

  template <typename T, typename R>
  typename std::enable_if<IsBuffer<R>::value, R>::type *mutable_data(
      MetalContext *context,
      size_t count,
      METAL_ACCESS_FLAG access = METAL_ACCESS_FLAG::CPUReadWrite) {
    target_ = TARGET(kMetal);
    buffer_->ResetLazyMetalBuffer<T>(context, count, access);
    dims_ = DDimLite({static_cast<int64_t>(count)});
    return static_cast<MetalBuffer *>(buffer_->data());
  }

  enum class MetalDataType : int {
    kRaw = 0,
    kMetal = 1,
  };
  MetalDataType metal_data_type_{MetalDataType::kRaw};
  MetalDataType metal_data_type() const { return metal_data_type_; }

  void *mutable_metal_data(void *ptr) {
    target_ = TARGET(kMetal);
    metal_data_type_ = MetalDataType::kMetal;
    buffer_->ResetLazyMetalData(ptr);
    return ptr;
  }
#endif

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data(TargetType target) {
#ifdef LITE_WITH_XPU
    if (target_ != target && target == TargetType::kXPU) {
      buffer_.reset(new XPUBuffer);
    }
#endif
    target_ = target;
    return mutable_data<T, R>();
  }

  template <typename T, typename R = T>
  R *mutable_data(TargetType target, size_t memory_size) {
#ifdef LITE_WITH_XPU
    if (target_ != target && target == TargetType::kXPU) {
      buffer_.reset(new XPUBuffer);
    }
#endif
    precision_ = lite_api::PrecisionTypeTrait<T>::Type();
    memory_size_ = memory_size;
    buffer_->ResetLazy(target, memory_size_);
    target_ = target;
    return reinterpret_cast<R *>(static_cast<char *>(buffer_->data()) +
                                 offset_);
  }

  void *mutable_data(size_t memory_size);
  void *mutable_data(TargetType target, size_t memory_size);

  const void *raw_data() const {
    return static_cast<char *>(
        (static_cast<char *>(buffer_->data()) + offset_));
  }

  void *raw_data() {
    return static_cast<char *>(
        (static_cast<char *>(buffer_->data()) + offset_));
  }

  void clear() {
    buffer_->Free();
    offset_ = 0;
  }
  size_t data_size() const { return this->dims().production(); }

  size_t memory_size() const { return memory_size_; }

  size_t offset() const { return offset_; }

  bool IsInitialized() const { return buffer_->data(); }

  // Other share data to this.
  void ShareDataWith(const TensorLite &other);

  void CopyDataFrom(const TensorLite &other);

  void ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size);

  TargetType target() const { return target_; }
  void set_target(TargetType target) { target_ = target; }

  template <typename T>
  TensorLite Slice(int64_t begin, int64_t end) const;

  friend STL::ostream &operator<<(STL::ostream &os, const TensorLite &tensor) {
    os << "Tensor:" << '\n';
    os << "dim: " << tensor.dims() << '\n';
    for (int i = 0; i < tensor.dims().production(); i++) {
      os << tensor.template data<float>()[i] << " ";
    }
    os << "\n";
    return os;
  }

 private:
  TargetType target_{TargetType::kHost};
  // precision_ and persistable_ are only used for persistable vars.
  // If your tensor wants to be saved and loaded correctly, you must
  // set values of precision_ and persistable_ after updating it.
  // If your tensor is just a temp tensor, such as activations,
  // you can ignore these two attributes.
  PrecisionType precision_{PrecisionType::kUnk};
  bool persistable_{false};

  DDimLite dims_;
  std::shared_ptr<Buffer> buffer_;
  LoD lod_;
  size_t memory_size_{};

  /// @brief Buffer may be shared with other tensors
  size_t offset_{0};
};

template <typename T>
TensorLite TensorLite::Slice(int64_t begin, int64_t end) const {
  CHECK_GE(begin, 0);
  CHECK_LE(end, dims_[0]);
  CHECK_LT(begin, end);
  if (dims_[0] == 1) {
    return *this;
  } else {
    int64_t base = numel() / dims_[0];
    TensorLite dst;
    dst.buffer_ = buffer_;
    dst.target_ = target_;
    auto dst_dims = dims_;
    dst_dims[0] = end - begin;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + static_cast<size_t>(begin * base) * sizeof(T);
    return dst;
  }
}

template <typename TensorT>
bool TensorCompareWith(const TensorT &a, const TensorT &b) {
  if (a.dims() != b.dims()) return false;
  if (a.precision() != b.precision()) return false;
  if (a.persistable() != b.persistable()) return false;
  if (memcmp(a.raw_data(), b.raw_data(), a.data_size()) != 0) return false;
  return true;
}

#ifdef LITE_WITH_OPENCL
template <>
const cl::Image2D *TensorLite::data<float, cl::Image2D>() const;

template <>  // use uint16_t represent half float
const cl::Image2D *TensorLite::data<uint16_t, cl::Image2D>() const;
#endif

#ifdef LITH_WITH_METAL
template <>
const metal_buffer *TensorLite::data<float, metal_buffer>() const;

template <>  // use uint16_t represent half float
const metal_buffer *TensorLite::data<uint16_t, metal_buffer>() const;

template <>
metal_buffer *TensorLite::mutable_data<float, metal_buffer>();

template <>  // use uint16_t represent half float
metal_buffer *TensorLite::mutable_data<uint16_t, metal_buffer>();

template <>
const metal_image *TensorLite::data<float, metal_image>() const;

template <>  // use uint16_t represent half float
const metal_image *TensorLite::data<uint16_t, metal_image>() const;

template <>
metal_image *TensorLite::mutable_data<float, metal_image>();

template <>  // use uint16_t represent half float
metal_image *TensorLite::mutable_data<uint16_t, metal_image>();
#endif

}  // namespace lite
}  // namespace paddle

#endif  // #ifndef LITE_WITH_FPGA
