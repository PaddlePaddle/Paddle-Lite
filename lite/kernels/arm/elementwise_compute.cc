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
#include "lite/kernels/host/elementwise_op_func.h"

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

inline bool is_fast_broadcast(const DDim& x_dims,
                              const DDim& y_dims,
                              int axis,
                              int* pre,
                              int* n,
                              int* post) {
  if (axis == -1) {
    axis = x_dims.size() - y_dims.size();
  }
  if (axis < 0) {
    LOG(INFO) << "Fast broadcast chk fail, for x_dims smaller.";
    return false;
  }
  DDim y_dim_trim = trim_trailing_singular_dims(y_dims);
  axis = (y_dim_trim.size() == 0) ? x_dims.size() : axis;
  if (x_dims.size() == y_dim_trim.size()) {
    LOG(INFO)
        << "Fast broadcast chk fail, for y's shape not really contained in x";
    return false;
  }
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dim_trim.size(); ++i) {
    if (x_dims[i + axis] != y_dim_trim[i]) {
      LOG(WARNING) << "Fast broadcast chk fail, for dimension mismatch.";
      return false;
    }
    (*n) *= y_dim_trim[i];
  }
  for (int i = axis + y_dim_trim.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
  return true;
}

template <class T>
using FastBCastFn = void(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <class T>
using ElementWiseFn = void(const T* dinx, const T* diny, T* dout, int num);

template <class T>
using BinaryOpFn = lite::kernels::host::BinaryOpFn<T>;

enum class OprandSwapable { NO, YES };

template <class Elem_t, class DimValue_t>
void common_elmentwise_op_arm(
    const lite::kernels::host::BatchElementWiseArg<Elem_t, DimValue_t>&
        batch_arg,
    BinaryOpFn<Elem_t> op,
    ElementWiseFn<Elem_t> elementwise_fn) {
  int batch_num = batch_arg.BatchNum();
  auto bcast_type = batch_arg.BcastType();
  int range_length = batch_arg.ElemNumPerBatch();
  switch (bcast_type) {
    case (lite::kernels::host::BroadcastType::X_AS_CONTINUOUS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        lite::kernels::host::element_wise_range_to_one<Elem_t>(
            batch_arg.XAtBatch(batch_id),
            batch_arg.YAtBatch(batch_id),
            batch_arg.ZAtBatch(batch_id),
            range_length,
            op);
      }
      break;
    }
    case (lite::kernels::host::BroadcastType::Y_AS_CONTINUOUS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        lite::kernels::host::element_wise_one_to_range<Elem_t>(
            batch_arg.XAtBatch(batch_id),
            batch_arg.YAtBatch(batch_id),
            batch_arg.ZAtBatch(batch_id),
            range_length,
            op);
      }
      break;
    }
    case (lite::kernels::host::BroadcastType::BOTH_CONTINUOUS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        elementwise_fn(batch_arg.XAtBatch(batch_id),
                       batch_arg.YAtBatch(batch_id),
                       batch_arg.ZAtBatch(batch_id),
                       range_length);
      }
      break;
    }
  }
}

template <class T, OprandSwapable opd_swap_able>
void elementwise_compute_template(paddle::lite::KernelBase* kernel,
                                  FastBCastFn<T> fast_bcast_fn,
                                  ElementWiseFn<T> elementwise_fn,
                                  BinaryOpFn<T> op) {
  auto& param = kernel->template Param<operators::ElementwiseParam>();
  auto* x_data = param.X->template data<T>();
  auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int pre, n, post;
  if (elementwise_fn && x_dims == y_dims) {
    elementwise_fn(x_data, y_data, out_data, x_dims.production());
  } else if (fast_bcast_fn &&
             is_fast_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(x_data, y_data, out_data, pre, n, post);
  } else if (fast_bcast_fn && opd_swap_able == OprandSwapable::YES &&
             axis == -1 &&
             is_fast_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(y_data, x_data, out_data, pre, n, post);
  } else if (elementwise_fn) {
    auto batch_arg = lite::kernels::host::GenBatchElementWiseArg<T>(
        param.X, param.Y, param.Out, axis);
    common_elmentwise_op_arm<T, int64_t>(batch_arg, op, elementwise_fn);
  }
  if (!elementwise_fn && !fast_bcast_fn) {
    LOG(FATAL) << "unsupported elementwise_compute called";
  }
}

template <typename T, PrecisionType PType>
void ElementwiseAddCompute<T, PType>::Run() {
  elementwise_compute_template<T, OprandSwapable::YES>(
      this,
      lite::arm::math::elementwise_add_broadcast<T>,
      lite::arm::math::elementwise_add<T>,
      paddle::lite::kernels::host::naive_add<T>);
}

void ElementwiseAddActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<float, OprandSwapable::YES>(
        this,
        lite::arm::math::elementwise_add_relu_broadcast<float>,
        lite::arm::math::elementwise_add_relu<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_add<float>,
            paddle::lite::kernels::host::naive_relu<float>>);
  }

  if (param.act_type == "tanh") {
    act_supported = true;
    elementwise_compute_template<float, OprandSwapable::YES>(
        this,
        nullptr,
        lite::arm::math::elementwise_add_tanh<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_add<float>,
            paddle::lite::kernels::host::naive_tanh<float>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

template <typename T, PrecisionType PType>
void ElementwiseSubCompute<T, PType>::Run() {
  elementwise_compute_template<T, OprandSwapable::NO>(
      this,
      lite::arm::math::elementwise_sub_broadcast<T>,
      lite::arm::math::elementwise_sub<T>,
      paddle::lite::kernels::host::naive_sub<T>);
}

void ElementwiseSubActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<float, OprandSwapable::NO>(
        this,
        lite::arm::math::elementwise_sub_relu_broadcast<float>,
        lite::arm::math::elementwise_sub_relu<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_sub<float>,
            paddle::lite::kernels::host::naive_relu<float>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

template <typename T, PrecisionType PType>
void ElementwiseMulCompute<T, PType>::Run() {
  elementwise_compute_template<T, OprandSwapable::YES>(
      this,
      lite::arm::math::elementwise_mul_broadcast<T>,
      lite::arm::math::elementwise_mul<T>,
      paddle::lite::kernels::host::naive_mul<T>);
}

template <typename T, PrecisionType PType>
void ElementwiseMulActivationCompute<T, PType>::Run() {
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<T, OprandSwapable::YES>(
        this,
        lite::arm::math::elementwise_mul_relu_broadcast<T>,
        lite::arm::math::elementwise_mul_relu<T>,
        paddle::lite::kernels::host::naive_fused_op<
            T,
            paddle::lite::kernels::host::naive_mul<T>,
            paddle::lite::kernels::host::naive_relu<T>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

void ElementwiseMaxCompute::Run() {
  elementwise_compute_template<float, OprandSwapable::YES>(
      this,
      lite::arm::math::elementwise_max_broadcast<float>,
      lite::arm::math::elementwise_max<float>,
      paddle::lite::kernels::host::naive_max<float>);
}

void ElementwiseMaxActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<float, OprandSwapable::YES>(
        this,
        lite::arm::math::elementwise_max_relu_broadcast<float>,
        lite::arm::math::elementwise_max_relu<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_max<float>,
            paddle::lite::kernels::host::naive_relu<float>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

template <typename T, PrecisionType PType>
void ElementwiseDivCompute<T, PType>::Run() {
  elementwise_compute_template<T, OprandSwapable::NO>(
      this,
      lite::arm::math::elementwise_div_broadcast<T>,
      lite::arm::math::elementwise_div<T>,
      paddle::lite::kernels::host::naive_div<T>);
}

void ElementwiseDivActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<float, OprandSwapable::NO>(
        this,
        lite::arm::math::elementwise_div_relu_broadcast<float>,
        lite::arm::math::elementwise_div_relu<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_div<float>,
            paddle::lite::kernels::host::naive_relu<float>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

template <typename T, PrecisionType PType>
void ElementwiseModCompute<T, PType>::Run() {
  elementwise_compute_template<T, OprandSwapable::NO>(
      this,
      lite::arm::math::elementwise_mod_broadcast<T>,
      lite::arm::math::elementwise_mod<T>,
      paddle::lite::kernels::host::naive_mod<T>);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using elementwise_add_float_t =
    paddle::lite::kernels::arm::ElementwiseAddCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kFloat, kNCHW, elementwise_add_float_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_add_int32_t =
    paddle::lite::kernels::arm::ElementwiseAddCompute<int32_t,
                                                      PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kInt32, kNCHW, elementwise_add_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
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

using elementwise_sub_float_t =
    paddle::lite::kernels::arm::ElementwiseSubCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_sub, kARM, kFloat, kNCHW, elementwise_sub_float_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_sub_int32_t =
    paddle::lite::kernels::arm::ElementwiseSubCompute<int32_t,
                                                      PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(
    elementwise_sub, kARM, kInt32, kNCHW, elementwise_sub_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
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

using elementwise_mul_int64_t =
    paddle::lite::kernels::arm::ElementwiseMulCompute<int64_t,
                                                      PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kInt64, kNCHW, elementwise_mul_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
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

using elementwise_div_int32_t =
    paddle::lite::kernels::arm::ElementwiseDivCompute<int32_t,
                                                      PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(
    elementwise_div, kARM, kInt32, kNCHW, elementwise_div_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
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
