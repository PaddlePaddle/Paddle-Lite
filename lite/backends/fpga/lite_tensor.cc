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

void TensorLite::ShareDataWith(const TensorLite &other) {
  dims_ = other.dims_;
  zynq_tensor_ = other.zynq_tensor_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  throw - 1;
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;

  if (zynq_tensor_.get() == nullptr) {
    zynq_tensor_.reset(new zynqmp::Tensor());
  }

  auto dt = zynq_tensor_->dataType();
  Resize(other.dims());
  auto shape = other.zynq_tensor_->shape();
  zynq_tensor_->mutableData<void>(zynq_tensor_->dataType(), shape);
  precision_ = other.precision_;

  memcpy(this->ZynqTensor()->data<void>(),
         other.ZynqTensor()->data<void>(),
         other.ZynqTensor()->shape().numel() * sizeof(float));
}

void *TensorLite::mutable_data(size_t memory_size) {
  memory_size_ = memory_size;

  std::vector<int> v_shape;
  for (int i = 0; i < dims_.size(); i++) {
    v_shape.push_back(dims_[i]);
  }
  zynqmp::LayoutType layout_type = get_layout_type(dims_);
  zynqmp::Shape input_shape(layout_type, v_shape);
  zynqmp::DataType data_type = precision_to_data_type(precision_);

  if (zynq_tensor_.get() == nullptr) {
    zynq_tensor_.reset(new zynqmp::Tensor());
  }
  return zynq_tensor_->mutableData<void>(data_type, input_shape);
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

zynqmp::LayoutType get_layout_type(DDimLite dims) {
  std::vector<int> v;
  for (int i = 0; i < dims.size(); i++) {
    v.push_back(dims[i]);
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
  return layout_type;
}

}  // namespace lite
}  // namespace paddle
