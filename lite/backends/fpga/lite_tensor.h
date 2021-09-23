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
#include "lite/core/dim.h"
#include "lite/core/memory.h"

namespace paddle {
namespace lite {

class DDimLite;
class TensorLite;

using DDim = lite::DDimLite;
using Tensor = lite::TensorLite;
using LoD = std::vector<std::vector<uint64_t>>;

// A light-weight tensor implementation.
class TensorLite {
 public:
  TensorLite() {}

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

  const void *raw_data() const { return zynq_tensor_->data<void>(); }

  void *raw_data() { return zynq_tensor_->data<void>(); }

  void clear() {
    if (zynq_tensor_) {
      memset(zynq_tensor_->data<void>(), 0, zynq_tensor_->memorySize());
    }
  }

  size_t data_size() const { return this->dims().production(); }

  size_t memory_size() const { return memory_size_; }

  size_t offset() const { return offset_; }

  bool IsInitialized() const { return zynq_tensor_ != nullptr; }

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
  LoD lod_;
  size_t memory_size_{};
  size_t offset_{0};

  std::shared_ptr<zynqmp::Tensor> zynq_tensor_;

  template <typename T>
  void mutable_data_internal();
};

inline zynqmp::DataType precision_to_data_type(PrecisionType p) {
  zynqmp::DataType data_type = zynqmp::FP32;
  switch (p) {
    case PrecisionType::kFloat:
      data_type = zynqmp::FP32;
      break;
    case PrecisionType::kFP16:
      data_type = zynqmp::FP16;
      break;
    case PrecisionType::kInt32:
      data_type = zynqmp::INT32;
      break;
    case PrecisionType::kInt16:
      data_type = zynqmp::INT16;
      break;
    case PrecisionType::kInt64:
      data_type = zynqmp::INT64;
      break;
    default:
      data_type = zynqmp::FP32;
      break;
  }
  return data_type;
}

template <typename T>
zynqmp::DataType get_data_type() {
  zynqmp::TypeResolver<T> resolver;
  return resolver();
}

zynqmp::LayoutType get_layout_type(DDimLite dims);

template <typename T, typename R>
R *TensorLite::mutable_data() {
  std::vector<int> v;
  for (int i = 0; i < dims_.size(); i++) {
    v.push_back(dims_[i]);
  }
  zynqmp::LayoutType layout_type = get_layout_type(dims_);
  zynqmp::Shape input_shape(layout_type, v);
  zynqmp::DataType data_type = get_data_type<T>();
  precision_ = lite_api::PrecisionTypeTrait<T>::Type();

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
  CHECK(dims_[0]);
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
    CHECK_GT(SIZE_MAX / sizeof(T), dst_dims.production());
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
  CHECK(dims_[0]);

  dst.target_ = target_;
  auto dst_dims = dims_;
  dst_dims[0] = end - begin;
  dst.Resize(dst_dims);
  void *dst_data = dst.mutable_data<T>();

  int64_t base = numel() / dims_[0];

  T *src_data = const_cast<T *>(data<T>());
  CHECK_GT(SIZE_MAX / sizeof(T), dst_dims.production());
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
