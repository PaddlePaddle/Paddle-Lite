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

#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "lite/backends/fpga/KD/tensor.hpp"
#include "lite/core/memory.h"

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

  friend std::ostream &operator<<(std::ostream &os, const DDimLite &dims) {
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
    return zynq_tensor_->data<R>() + offset_;
    // return zynq_tensor_->data<R>();
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
  R *mutable_data();

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data(TargetType target);
  void *mutable_data(size_t memory_size);
  void *mutable_data(TargetType target, size_t memory_size);

  const void *raw_data() const {
    return buffer_->data();
  }  // TODO(chonwhite) delete buffer;

  void clear() {
    // zynq_tensor_->releaseData();
    if (zynq_tensor_) {
      memset(zynq_tensor_->data<void>(), 0, zynq_tensor_->memorySize());
    }
  }

  size_t data_size() const { return this->dims().production(); }

  size_t memory_size() const { return zynq_tensor_->memorySize(); }

  size_t offset() const { return offset_; }

  bool IsInitialized() const {
    return buffer_->data();
  }  // TODO(chonwhite) delete buffer;

  // Other share data to this.
  void ShareDataWith(const TensorLite &other);

  void CopyDataFrom(const TensorLite &other);

  void ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size) {
    // TODO(chonwhite) deal with buffer;
  }

  template <typename T>
  TensorLite Slice(int64_t begin, int64_t end) const;

  template <typename T>
  void Slice(TensorLite &dst, int64_t begin, int64_t end) const;  // NOLINT

  TargetType target() const { return target_; }

  zynqmp::Tensor *ZynqTensor() const { return zynq_tensor_.get(); }

  friend std::ostream &operator<<(std::ostream &os, const TensorLite &tensor) {
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
  PrecisionType precision_{PrecisionType::kFloat};
  bool persistable_{false};

  DDimLite dims_;
  std::shared_ptr<Buffer> buffer_;
  LoD lod_;
  size_t memory_size_{};
  size_t offset_{0};

  std::shared_ptr<zynqmp::Tensor> zynq_tensor_;

  template <typename T>
  void mutable_data_internal();
};

template <typename T>
zynqmp::DataType get_date_type() {
  zynqmp::DataType data_type = zynqmp::FP32;
  if (typeid(T) == typeid(float)) {
    data_type = zynqmp::FP32;
  }
  if (typeid(T) == typeid(zynqmp::float16)) {
    data_type = zynqmp::FP16;
  }
  if (typeid(T) == typeid(int)) {
    data_type = zynqmp::INT32;
  }
  if (typeid(T) == typeid(int32_t)) {
    data_type = zynqmp::INT32;
  }
  if (typeid(T) == typeid(int8_t)) {
    data_type = zynqmp::INT8;
  }
  if (typeid(T) == typeid(int64_t)) {
    data_type = zynqmp::INT64;
  }

  return data_type;
}

template <typename T>
PrecisionType get_precistion_type() {
  PrecisionType data_type = PrecisionType::kUnk;
  if (typeid(T) == typeid(float)) {
    data_type = PrecisionType::kFloat;
  }
  if (typeid(T) == typeid(zynqmp::float16)) {
    data_type = PrecisionType::kFP16;
  }
  if (typeid(T) == typeid(int)) {
    data_type = PrecisionType::kInt32;
  }
  if (typeid(T) == typeid(int32_t)) {
    data_type = PrecisionType::kInt32;
  }
  if (typeid(T) == typeid(int8_t)) {
    data_type = PrecisionType::kInt8;
  }
  if (typeid(T) == typeid(int64_t)) {
    data_type = PrecisionType::kInt64;
  }

  return data_type;
}

template <typename T, typename R>
R *TensorLite::mutable_data() {
  std::vector<int> v;
  for (int i = 0; i < dims_.size(); i++) {
    v.push_back(dims_[i]);
  }
  zynqmp::LayoutType layout_type = zynqmp::NCHW;
  switch (v.size()) {
    case 0:
      layout_type = zynqmp::None;
      break;
    case 1:
      layout_type = zynqmp::N;
      break;
    case 2:
      layout_type = zynqmp::NC;
      break;
    case 3:
      layout_type = zynqmp::NHW;
      break;
    case 4:
      layout_type = zynqmp::NCHW;
      break;
  }
  zynqmp::Shape input_shape(layout_type, v);
  zynqmp::DataType data_type = get_date_type<T>();
  precision_ = get_precistion_type<T>();

  if (zynq_tensor_.get() == nullptr) {
    zynq_tensor_.reset(new zynqmp::Tensor());
  }

  return zynq_tensor_->mutableData<R>(data_type, input_shape);
}

template <typename T, typename R>
R *TensorLite::mutable_data(TargetType target) {
  target_ = target;
  return mutable_data<T>();
}

template <typename T>
TensorLite TensorLite::Slice(int64_t begin, int64_t end) const {
  throw - 1;
  CHECK_GE(begin, 0);
  CHECK_LE(end, dims_[0]);
  CHECK_LT(begin, end);
  if (dims_[0] == 1) {
    return *this;
  } else {
    int64_t base = numel() / dims_[0];

    TensorLite dst;
    dst.target_ = target_;
    auto dst_dims = dims_;
    dst_dims[0] = end - begin;
    dst.Resize(dst_dims);
    void *dst_data = dst.mutable_data<T>();

    T *src_data = const_cast<T *>(data<T>());
    memcpy(dst_data,
           src_data + static_cast<size_t>(begin * base) * sizeof(T),
           dst_dims.production() * sizeof(T));
    return dst;
  }
}

template <typename T>
void TensorLite::Slice(TensorLite &dst, int64_t begin, int64_t end) const {
  // TODO(chonwhite) delete this function;
  CHECK_GE(begin, 0);
  CHECK_LE(end, dims_[0]);
  CHECK_LT(begin, end);

  dst.target_ = target_;
  auto dst_dims = dims_;
  dst_dims[0] = end - begin;
  dst.Resize(dst_dims);
  void *dst_data = dst.mutable_data<T>();

  int64_t base = numel() / dims_[0];

  T *src_data = const_cast<T *>(data<T>());
  memcpy(dst_data,
         src_data + static_cast<size_t>(begin * dst_dims.production()),
         dst_dims.production() * sizeof(T));
}

template <typename TensorT>
bool TensorCompareWith(const TensorT &a, const TensorT &b) {
  if (a.dims() != b.dims()) return false;
  if (memcmp(a.raw_data(), b.raw_data(), a.data_size()) != 0) return false;
  return true;
}

}  // namespace lite
}  // namespace paddle
