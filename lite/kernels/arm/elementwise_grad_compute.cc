// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/elementwise_grad_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

inline DDim trim_trailing_singular_dims(const DDim& dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }

  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim();
  }
  return DDim(trim_dims);
}

inline bool is_broadcast(const DDim& x_dims,
                         const DDim& y_dims,
                         int axis,
                         int* pre,
                         int* n,
                         int* post) {
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  DDim y_dim_trim = trim_trailing_singular_dims(y_dims);
  axis = (y_dim_trim.size() == 0) ? x_dims.size() : axis;
  if (x_dims.size() == y_dim_trim.size()) {
    return false;
  }
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dim_trim.size(); ++i) {
    CHECK_EQ(x_dims[i + axis], y_dim_trim[i])
        << "Broadcast dimension mismatch.";
    (*n) *= y_dim_trim[i];
  }
  for (int i = axis + y_dim_trim.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
  return true;
}

void ElementwiseAddGradCompute::Run() {
  auto& param = Param<operators::ElementwiseGradParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  const float* out_grad_data = param.OutGrad->data<float>();
  float* x_grad_data = nullptr;
  float* y_grad_data = nullptr;
  if (param.XGrad) {
    x_grad_data = param.XGrad->mutable_data<float>();
  }
  if (param.YGrad) {
    y_grad_data = param.YGrad->mutable_data<float>();
  }
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (!param.XGrad) {
    CHECK(param.YGrad);
    lite::arm::math::elementwise_add_grad(
        out_grad_data, y_grad_data, y_dims.production());
    return;
  }

  if (!param.YGrad) {
    CHECK(param.XGrad);
    lite::arm::math::elementwise_add_grad(
        out_grad_data, x_grad_data, x_dims.production());
    return;
  }

  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_add_grad_broadcast(
        out_grad_data, y_grad_data, x_grad_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_add_grad_broadcast(
        out_grad_data, x_grad_data, y_grad_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_add_grad(
        out_grad_data, x_grad_data, x_dims.production());
    lite::arm::math::elementwise_add_grad(
        out_grad_data, y_grad_data, y_dims.production());
  }
}

void ElementwiseSubGradCompute::Run() {
  auto& param = Param<operators::ElementwiseGradParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  const float* out_data = param.OutGrad->data<float>();
  float* x_grad_data = nullptr;
  float* y_grad_data = nullptr;
  if (param.XGrad) {
    x_grad_data = param.XGrad->mutable_data<float>();
  }
  if (param.YGrad) {
    y_grad_data = param.YGrad->mutable_data<float>();
  }
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;

  if (!param.XGrad || !param.YGrad) {
    CHECK(param.XGrad || param.YGrad);
    if (param.XGrad) {
      lite::arm::math::elementwise_sub_grad(
          out_data, x_grad_data, y_grad_data, x_dims.production());
      return;
    } else {
      lite::arm::math::elementwise_sub_grad(
          out_data, x_grad_data, y_grad_data, y_dims.production());
      return;
    }
  }

  if (x_dims.size() < y_dims.size()) {
    LOG(FATAL) << "elewise sub grad don't support x_dims size < y_dims size";
  }
  if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_sub_grad_broadcast(
        out_data, x_grad_data, y_grad_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_sub_grad(
        out_data, x_grad_data, y_grad_data, x_dims.production());
  }
}

template <typename T, PrecisionType PType>
void ElementwiseMulGradCompute<T, PType>::Run() {
  LOG(FATAL) << "elementwise mul_grad not implement yet";
}

void ElementwiseMaxGradCompute::Run() {
  LOG(FATAL) << "elementwise max_grad not implement yet";
}

void ElementwiseDivGradCompute::Run() {
  LOG(FATAL) << "elementwise div_grad not implement yet";
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using elementwise_mul_grad_float =
    paddle::lite::kernels::arm::ElementwiseMulGradCompute<float,
                                                          PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(elementwise_add_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseAddGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseSubGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseDivGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_mul_grad, kARM, kFloat, kNCHW, elementwise_mul_grad_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseMaxGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
