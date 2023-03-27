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
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

class IntArray {
 public:
  IntArray() = default;
  IntArray(const std::vector<int32_t>& val)  // NOLINT
      : array_(val.begin(), val.end()) {}

  IntArray(const std::vector<int64_t>& val) : array_(val) {}  // NOLINT

  IntArray(std::initializer_list<int64_t> val) : array_(val) {}

  IntArray(const int64_t* date_value, int64_t n) { AssignData(date_value, n); }

  IntArray(const int32_t* date_value, int64_t n) { AssignData(date_value, n); }

  IntArray(const IntArray& other)
      : array_(other.array_),
        tensors_(other.tensors_),
        is_multi_tensor_(other.is_multi_tensor_) {}

  IntArray(const lite::Tensor* tensor) {  // NOLINT
    tensors_.push_back(tensor);
  }

  IntArray(const std::vector<const lite::Tensor*>& tensors)  // NOLINT
      : tensors_(tensors),
        is_multi_tensor_(true) {}

  IntArray(const std::vector<lite::Tensor*>& tensors)  // NOLINT
      : tensors_(tensors.begin(), tensors.end()),
        is_multi_tensor_(true) {}

  const std::vector<int64_t>& GetData() const {
    if (FromTensor()) {
      if (!is_multi_tensor_) {
        AssignDataFromTensor(*tensors_[0]);
      } else {
        AssignDataFromTensors(this->tensors_);
      }
    }
    return array_;
  }

  size_t size() const { return array_.size(); }

  bool FromTensor() const { return !tensors_.empty(); }

 private:
  template <typename T>
  void AssignData(const T* data, int64_t n) const {
    if (data || n == 0) {
      array_.reserve(n);
      for (auto i = 0; i < n; ++i) {
        array_.push_back(static_cast<int64_t>(data[i]));
      }
    } else {
      LOG(FATAL) << "The input data pointer is null.";
    }
  }

  void AssignDataFromTensor(const lite::Tensor& tensor) const {
    size_t n = tensor.numel();
    array_.clear();
    switch (tensor.precision()) {
      case PrecisionType::kInt32:
        AssignData(tensor.data<int32_t>(), n);
        break;
      case PrecisionType::kInt64:
        AssignData(tensor.data<int64_t>(), n);
        break;
      default:
        LOG(FATAL) << "Unsupported data type: the type of IntArray "
                      "should be int32/int64, but received "
                   << static_cast<int>(tensor.precision());
    }
  }

  void AssignDataFromTensors(
      const std::vector<const lite::Tensor*>& tensors) const {
    array_.clear();
    array_.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto precision_type = tensors[i]->precision();
      switch (precision_type) {
        case PrecisionType::kInt32:
          array_.push_back(tensors[i]->data<int32_t>()[0]);
          break;
        case PrecisionType::kInt64:
          array_.push_back(tensors[i]->data<int64_t>()[0]);
          break;
        default:
          LOG(FATAL) << "Unsupported data type: the type of IntArray "
                        "should be int32/int64, but received "
                     << static_cast<int>(precision_type);
      }
    }
  }

 private:
  mutable std::vector<int64_t> array_;
  std::vector<const lite::Tensor*> tensors_;
  bool is_multi_tensor_{false};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
