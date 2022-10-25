// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

class Scalar {
 public:
  Scalar() : Scalar(0) {}
  Scalar(bool val) : dtype_(PrecisionType::kBool) {  // NOLINT
    data_.b = val;
  }

  Scalar(int32_t val) : dtype_(PrecisionType::kInt32) {  // NOLINT
    data_.i32 = val;
  }

  Scalar(int64_t val) : dtype_(PrecisionType::kInt64) {  // NOLINT
    data_.i64 = val;
  }

  Scalar(float val) : dtype_(PrecisionType::kFloat) {  // NOLINT
    data_.f32 = val;
  }

  Scalar(double val) : dtype_(PrecisionType::kFP64) {  // NOLINT
    data_.f64 = val;
  }

  Scalar(const Scalar& other) {  // NOLINT
    this->dtype_ = other.dtype_;
    this->data_.f64 = other.data_.f64;
    this->tensor_ = other.tensor_;
  }

  // construct from a tensor
  Scalar(const lite::Tensor* tensor)  // NOLINT
      : tensor_(tensor),
        dtype_(tensor->precision()) {}

  template <typename RT>
  inline RT to() const {
    if (FromTensor()) {
      GetDataFromTensor(*tensor_);
    }
    switch (dtype_) {
      case PrecisionType::kBool:
        return static_cast<RT>(data_.b);
      case PrecisionType::kFloat:
        return static_cast<RT>(data_.f32);
      case PrecisionType::kFP64:
        return static_cast<RT>(data_.f64);
      case PrecisionType::kInt32:
        return static_cast<RT>(data_.i32);
      case PrecisionType::kInt64:
        return static_cast<RT>(data_.i64);
      default:
        LOG(FATAL) << "Unsupported data type: the type of Scalar "
                      "should be bool/int/float/double, but received "
                   << static_cast<int>(dtype_);
        return static_cast<RT>(0);
    }
  }

  bool FromTensor() const { return tensor_ != nullptr; }

  void SetTensor(Tensor* tensor) {
    tensor_ = tensor;
    dtype_ = tensor->precision();
  }

  PrecisionType dtype() const { return dtype_; }

 private:
  void GetDataFromTensor(const lite::Tensor& tensor) const {
    switch (dtype_) {
      case PrecisionType::kBool:
        data_.b = tensor.template data<bool>()[0];
        break;
      case PrecisionType::kFloat:
        data_.f32 = tensor.template data<float>()[0];
        break;
      case PrecisionType::kFP64:
        data_.f64 = tensor.template data<double>()[0];
        break;
      case PrecisionType::kInt32:
        data_.i32 = tensor.template data<int32_t>()[0];
        break;
      case PrecisionType::kInt64:
        data_.i64 = tensor.template data<int64_t>()[0];
        break;
      default:
        LOG(FATAL) << "Unsupported data type: the type of Scalar "
                      "should be bool/int/float/double, but received "
                   << static_cast<int>(dtype_);
    }
  }

 private:
  const Tensor* tensor_{nullptr};

  PrecisionType dtype_;
  mutable union data {
    bool b;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
  } data_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
