/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef RESHAPE_OP

#pragma once

#include <vector>
#include "framework/operator.h"

#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

inline framework::DDim ValidateShape(const std::vector<int> shape,
                                     const framework::DDim& in_dims) {
  const int64_t in_size = framework::product(in_dims);
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_MOBILE_ENFORCE(
          unk_dim_idx == -1,
          "Only one input dimension of Attr(shape) can be unknown.");
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_MOBILE_ENFORCE(
          static_cast<int>(i) < in_dims.size(),
          "The index of dimension to copy from input shape must be less "
          "than the size of input shape.");
    } else {
      PADDLE_MOBILE_ENFORCE(
          shape[i] > 0,
          "Each input dimension of Attr(shape) must not be negtive except "
          "one unknown dimension.");
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    output_shape[unk_dim_idx] = -in_size / capacity;
    PADDLE_MOBILE_ENFORCE(output_shape[unk_dim_idx] * capacity == -in_size,
                          "Invalid shape is given.");
  } else {
    PADDLE_MOBILE_ENFORCE(capacity == in_size, "Invalid shape is given.");
  }
  return framework::make_ddim(output_shape);
}

template <typename DeviceType, typename T>
class ReshapeKernel
    : public framework::OpKernelBase<DeviceType, ReshapeParam<DeviceType>> {
 public:
  void Compute(const ReshapeParam<DeviceType>& param);
  bool Init(ReshapeParam<DeviceType>* param);
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
