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
  buffer_ = other.buffer_;  // TODO(chonwhite) delete buffer;
  dims_ = other.dims_;
  zynq_tensor_ = other.zynq_tensor_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  throw - 1;
}

void *TensorLite::mutable_data(size_t memory_size) {
  memory_size_ = memory_size;  // TODO(chonwhite) delete buffer;
  buffer_->ResetLazy(target_, memory_size_);
  // throw -1;
  // std::cout << memory_size << std::endl;
  return buffer_->data();
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  // std::cout << "other11:: "<< &other << std::endl;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  // std::cout << "before dataType\n";

  if (zynq_tensor_.get() == nullptr) {
    zynq_tensor_.reset(new zynqmp::Tensor());
  }

  auto dt = zynq_tensor_->dataType();
  // std::cout << "after dataType\n";

  // std::cout << "before resize\n";
  Resize(other.dims());
  auto shape = other.zynq_tensor_->shape();
  // std::cout << "after resize\n";
  zynq_tensor_->mutableData<void>(zynq_tensor_->dataType(), shape);
  // std::cout << "after mutableData\n";
  // std::cout << "ZynqTensor():" << this->ZynqTensor() << std::endl;
  // std::cout << "other Tensor():" << other.ZynqTensor() << std::endl;

  // this->ZynqTensor()->copyFrom(other.ZynqTensor());
  memcpy(this->ZynqTensor()->data<void>(),
         other.ZynqTensor()->data<void>(),
         other.ZynqTensor()->shape().numel() * sizeof(float));
  // memcpy()
  // std::cout << "after copyFrom\n";
}

}  // namespace lite
}  // namespace paddle
