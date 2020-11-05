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

#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "lite/core/memory.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {

class DDimLite;
class TensorLite;

using DDim = lite::DDimLite;
using Tensor = lite::TensorLite;

class DDimLite {
 public:
  using value_type = int64_t;

  DDimLite() = default;

  explicit DDimLite(const std::vector<value_type> &x) { ConstructFrom(x); }
  // DDimLite(std::initializer_list<value_type> init_list) :
  // DDimLite(std::vector<value_type>(init_list)) {}

  void ConstructFrom(const std::vector<value_type> &x) { data_ = x; }

  value_type operator[](int offset) const { return data_[offset]; }
  value_type &operator[](int offset) { return data_[offset]; }
  std::vector<int64_t> Vectorize() const { return data_; }

  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  value_type production() const;

  const std::vector<value_type> &data() const { return data_; }
  value_type count(int start, int end) const;

  DDimLite Slice(int start, int end) const;

  DDimLite Flatten2D(int col) const {
    return DDimLite(std::vector<value_type>(
        {Slice(0, col).production(), Slice(col, size()).production()}));
  }

  std::string repr() const;

  friend STL::ostream &operator<<(STL::ostream &os, const DDimLite &dims) {
    os << dims.repr();
    return os;
  }

  friend bool operator==(const DDimLite &a, const DDimLite &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const DDimLite &a, const DDimLite &b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return true;
    }
    return false;
  }

 private:
  std::vector<value_type> data_;
};

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

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data(TargetType target) {
    target_ = target;
    return mutable_data<T, R>();
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

}  // namespace lite
}  // namespace paddle

#endif  // #ifndef LITE_WITH_FPGA
