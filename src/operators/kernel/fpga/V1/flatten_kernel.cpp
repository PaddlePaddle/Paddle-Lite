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

#ifdef FLATTEN_OP

#include "operators/kernel/flatten_kernel.h"
#include "operators/kernel/central-arm-func/flatten_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FlattenKernel<FPGA, float>::Init(FlattenParam<FPGA> *param) {
  return true;
}

template <>
void FlattenKernel<FPGA, float>::Compute(const FlattenParam<FPGA> &param) {
  const auto *input_x = param.InputX();
  const auto axis = param.Axis();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();

  const auto &out_shape_v = GetOutputShape(axis, input_x_dims);
  const framework::DDim &out_dim = ValidateShape(out_shape_v, input_x_dims);

  out->Resize(out_dim);
  out->mutable_data<float>();
  // framework::TensorCopy(*input_x, out);

  input_x->check_memory_size();
  out->Resize(input_x->dims());
  auto src_ptr = input_x->data<void>();
  auto dst_ptr = out->mutable_data(input_x->type());
  auto size = input_x->numel() * sizeof(half);
  memory::Copy(dst_ptr, src_ptr, size);

  out->Resize(out_dim);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
