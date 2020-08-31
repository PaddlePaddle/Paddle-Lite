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

#include "lite/kernels/arm/elementwise_compute.h"
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

void ElementwiseAddCompute::Run() {
  auto& param = Param<operators::ElementwiseParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_add_broadcast(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_add_broadcast(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_add(
        x_data, y_data, out_data, x_dims.production());
  }
}

void ElementwiseAddActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  std::string act_type = param.act_type;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_add_relu_broadcast(
          y_data, x_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_add_relu_broadcast(
          x_data, y_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else {
    if (act_type == "relu") {
      lite::arm::math::elementwise_add_relu(
          x_data, y_data, out_data, x_dims.production());
    } else if (act_type == "tanh") {
      lite::arm::math::elementwise_add_tanh(
          x_data, y_data, out_data, x_dims.production());
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  }
}

void ElementwiseSubCompute::Run() {
  auto& param = Param<operators::ElementwiseParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_sub_broadcast(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_sub_broadcast(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_sub(
        x_data, y_data, out_data, x_dims.production());
  }
}

void ElementwiseSubActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  std::string act_type = param.act_type;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;

  if (act_type != "relu") {
    LOG(FATAL) << "unsupported Activation type: " << act_type;
  }
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_sub_relu_broadcast(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_sub_relu_broadcast(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_sub_relu(
        x_data, y_data, out_data, x_dims.production());
  }
}

template <typename T, PrecisionType PType>
void ElementwiseMulCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ElementwiseParam>();
  auto* x_data = param.X->template data<T>();
  auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_mul_broadcast<T>(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_mul_broadcast<T>(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_mul<T>(
        x_data, y_data, out_data, x_dims.production());
  }
}

template <typename T, PrecisionType PType>
void ElementwiseMulActivationCompute<T, PType>::Run() {
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  auto* x_data = param.X->template data<T>();
  auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  std::string act_type = param.act_type;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_mul_relu_broadcast<T>(
          y_data, x_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_mul_relu_broadcast<T>(
          x_data, y_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else {
    if (act_type == "relu") {
      lite::arm::math::elementwise_mul_relu<T>(
          x_data, y_data, out_data, x_dims.production());
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  }
}

void ElementwiseMaxCompute::Run() {
  auto& param = Param<operators::ElementwiseParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_max_broadcast(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_max_broadcast(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_max(
        x_data, y_data, out_data, x_dims.production());
  }
}

void ElementwiseMaxActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  std::string act_type = param.act_type;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_max_relu_broadcast<float>(
          y_data, x_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_max_relu_broadcast(
          x_data, y_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else {
    if (act_type == "relu") {
      lite::arm::math::elementwise_max_relu(
          x_data, y_data, out_data, x_dims.production());
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  }
}

template <typename T, PrecisionType PType>
void ElementwiseDivCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ElementwiseParam>();
  auto* x_data = param.X->template data<T>();
  auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size()) {
    LOG(FATAL) << "elewise div don't support x_dims size < y_dims size";
  }
  if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_div_broadcast<T>(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_div<T>(
        x_data, y_data, out_data, x_dims.production());
  }
}

void ElementwiseDivActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  std::string act_type = param.act_type;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  if (x_dims.size() < y_dims.size()) {
    LOG(FATAL) << "elewise div don't support x_dims size < y_dims size";
  }
  int pre, n, post;
  if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    if (act_type == "relu") {
      lite::arm::math::elementwise_div_relu_broadcast(
          x_data, y_data, out_data, pre, n, post);
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  } else {
    if (act_type == "relu") {
      lite::arm::math::elementwise_div_relu(
          x_data, y_data, out_data, x_dims.production());
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type;
    }
  }
}

template <typename T, PrecisionType PType>
void ElementwiseModCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ElementwiseParam>();
  auto* x_data = param.X->template data<T>();
  auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (x_dims.size() < y_dims.size() &&
      is_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_mod_broadcast<T>(
        y_data, x_data, out_data, pre, n, post);
  } else if (is_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    lite::arm::math::elementwise_mod_broadcast<T>(
        x_data, y_data, out_data, pre, n, post);
  } else {
    lite::arm::math::elementwise_mod<T>(
        x_data, y_data, out_data, x_dims.production());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::ElementwiseAddActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseSubCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_sub_activation,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::ElementwiseSubActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_mul_float_t =
    paddle::lite::kernels::arm::ElementwiseMulCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kFloat, kNCHW, elementwise_mul_float_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_mul_int32_t =
    paddle::lite::kernels::arm::ElementwiseMulCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kInt32, kNCHW, elementwise_mul_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_mul_int64_t =
    paddle::lite::kernels::arm::ElementwiseMulCompute<int64_t,
                                                      PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kInt64, kNCHW, elementwise_mul_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using fusion_elementwise_mul_activation_float_t = paddle::lite::kernels::arm::
    ElementwiseMulActivationCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(fusion_elementwise_mul_activation,
                     kARM,
                     kFloat,
                     kNCHW,
                     fusion_elementwise_mul_activation_float_t,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using fusion_elementwise_mul_activation_int64_t = paddle::lite::kernels::arm::
    ElementwiseMulActivationCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(fusion_elementwise_mul_activation,
                     kARM,
                     kInt64,
                     kNCHW,
                     fusion_elementwise_mul_activation_int64_t,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseMaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_max_activation,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::ElementwiseMaxActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_div_fp32_t =
    paddle::lite::kernels::arm::ElementwiseDivCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    elementwise_div, kARM, kFloat, kNCHW, elementwise_div_fp32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_div_int64_t =
    paddle::lite::kernels::arm::ElementwiseDivCompute<int64_t,
                                                      PRECISION(kInt64)>;

REGISTER_LITE_KERNEL(
    elementwise_div, kARM, kInt64, kNCHW, elementwise_div_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_div_activation,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::ElementwiseDivActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_mod_int64_t =
    paddle::lite::kernels::arm::ElementwiseModCompute<int64_t,
                                                      PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(
    elementwise_mod, kARM, kInt64, kNCHW, elementwise_mod_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
