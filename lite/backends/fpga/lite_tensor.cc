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

#include "lite/backends/fpga/lite_tensor.h"
#include <string>

namespace paddle {
namespace lite {

using value_type = int64_t;

value_type DDimLite::production() const {
  value_type res = 1;
  for (size_t i = 0; i < this->size(); i++) {
    res *= (*this)[i];
  }
  return res;
}

value_type DDimLite::count(int start, int end) const {
  if (start < 0) {
    start = 0;
  }
  if (end > size()) {
    end = size();
  }
  if (end < start) {
    end = start;
  }
  value_type sum = 1;
  for (auto i = start; i < end; ++i) {
    sum *= data_[i];
  }
  return sum;
}

DDimLite DDimLite::Slice(int start, int end) const {
  std::vector<value_type> vec;
  for (int i = start; i < end; i++) {
    vec.push_back((*this)[i]);
  }
  return DDimLite(vec);
}

std::string DDimLite::repr() const {
  std::stringstream ss;
  if (empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < this->size() - 1; i++) {
    ss << (*this)[i] << ",";
  }
  if (!this->empty()) ss << (*this)[size() - 1];
  ss << "}";
  return ss.str();
}

void TensorLite::ShareDataWith(const TensorLite &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  zynq_tensor_ = other.zynq_tensor_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  throw - 1;
}

void *TensorLite::mutable_data(size_t memory_size) {
  memory_size_ = memory_size;
  buffer_->ResetLazy(target_, memory_size_);
  // throw -1;
  std::cout << memory_size << std::endl;
  return buffer_->data();
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  // memory_size_ = other.memory_size_;
  // buffer_->CopyDataFrom(*other.buffer_, memory_size_);
  zynq_tensor_->mutableData<void>(other.zynq_tensor_->dataType(),
                                  other.zynq_tensor_->shape());
}

// template <typename T>
// void TensorLite::mutable_data_internal() {

// }

}  // namespace lite
}  // namespace paddle
