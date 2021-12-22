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

// Attention! There is no guarantee that dividing(or floordividing)
// by 0 will get the correct result in ElementWise OP.

#include "lite/kernels/x86/elementwise_compute.h"
#include <string>
#include <vector>
#include "lite/backends/x86/math/elementwise.h"
#include "lite/backends/x86/math/elementwise_common_broadcast_config.h"
#include "lite/kernels/host/elementwise_op_func.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

namespace x86_math = paddle::lite::x86::math;

// Remove trailing dimensions of size 1 for y
DDim trim_trailing_singular_dims(const DDim& dims) {
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

/*
 * Out = X point dot Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 * 3. force x_dims.size() is greater than y_dims.size(), else
 *    return false.
 */
bool is_fast_broadcast(const DDim& x_dims,
                       const DDim& y_dims,
                       int axis,
                       int* pre,
                       int* n,
                       int* post) {
  if (axis == -1) {
    axis = x_dims.size() - y_dims.size();
  }
  if (axis < 0) {
    VLOG(4) << "Fast broadcast chk fail, for x_dims smaller.";
    return false;
  }
  DDim y_dim_trim = trim_trailing_singular_dims(y_dims);
  axis = (y_dim_trim.size() == 0) ? x_dims.size() : axis;
  if (x_dims.size() < (y_dim_trim.size() + axis)) {
    VLOG(4) << "Fast broadcast chk fail, for y's shape size doesnt follow the "
               "axis rule";
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
      VLOG(4) << "Fast broadcast chk fail, for dimension mismatch.";
      return false;
    }
    (*n) *= y_dim_trim[i];
  }
  for (int i = axis + y_dim_trim.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
  return true;
}

// function pointer
template <class T>
using FastBCastFn = void(const T* dinx,
                         const T* diny,
                         T* dout,
                         int batch,
                         int channels,
                         int num,
                         bool has_active,
                         std::string act_mode,
                         bool inv);

template <class T>
using ElementWiseFn = void(const T* dinx,
                           const T* diny,
                           T* dout,
                           int num,
                           bool has_active,
                           std::string act_mode);

template <class T>
using BinaryOpFn = lite::kernels::host::BinaryOpFn<T>;

template <class Elem_t, class DimValue_t, class X86Config>
struct X86CommonElementWise {
  static void Run(
      // todo: if necessary, generate
      //  lite::kernels::host::StaticBatchElementWiseArg by
      //  batch_arg->ToStaticArg() before kernel launch, it will help to reduce
      //  runtime overhead.
      const lite::kernels::host::BatchElementWiseArg<Elem_t, DimValue_t>&
          batch_arg,
      BinaryOpFn<Elem_t> op) {
    int batch_num = batch_arg.BatchNum();
    auto bcast_type = batch_arg.BcastType();
    int range_length = batch_arg.ElemNumPerBatch();
    switch (bcast_type) {
      case (lite::kernels::host::BroadcastType::X_AS_CONTINUOUS): {
        for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
          paddle::lite::x86::math::elementwise_range_to_one<X86Config>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      case (lite::kernels::host::BroadcastType::Y_AS_CONTINUOUS): {
        for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
          paddle::lite::x86::math::elementwise_one_to_range<X86Config>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      case (lite::kernels::host::BroadcastType::BOTH_CONTINUOUS): {
        for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
          paddle::lite::x86::math::elementwise_range_to_range<X86Config>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      default: {
        LOG(FATAL) << "Un supported bcast type(isa)";
        break;
      }
    }
  }
};

template <class Elem_t, class DimValue_t>
struct X86CommonElementWise<Elem_t,
                            DimValue_t,
                            paddle::lite::x86::math::NullCpuInstruction> {
  static void Run(
      // todo: if necessary, generate
      //  lite::kernels::host::StaticBatchElementWiseArg by
      //  batch_arg->ToStaticArg() before kernel launch, it will help to reduce
      //  runtime overhead.
      const lite::kernels::host::BatchElementWiseArg<Elem_t, DimValue_t>&
          batch_arg,
      BinaryOpFn<Elem_t> op) {
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
          lite::kernels::host::element_wise_range_to_range<Elem_t>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length,
              op);
        }
        break;
      }
      default: {
        LOG(FATAL) << "Un supported bcast type(host)";
        break;
      }
    }
  }
};

template <class OpParamType, class T, class X86Config>
void elementwise_compute_template(paddle::lite::KernelBase* kernel,
                                  FastBCastFn<T> fast_bcast_fn,
                                  ElementWiseFn<T> elementwise_fn,
                                  BinaryOpFn<T> op,
                                  bool has_active = false,
                                  std::string act_type = "") {
  auto& param = kernel->template Param<OpParamType>();
  auto x = param.X;
  auto y = param.Y;

  auto* x_data = x->template data<T>();
  auto* y_data = y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  int axis = param.axis;
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  int pre, n, post;

  if (elementwise_fn && x_dims == y_dims) {
    elementwise_fn(
        x_data, y_data, out_data, x_dims.production(), has_active, act_type);
  } else if (fast_bcast_fn &&
             is_fast_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(
        x_data, y_data, out_data, pre, n, post, has_active, act_type, false);
  } else if (fast_bcast_fn && axis == -1 &&
             is_fast_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(
        x_data, y_data, out_data, pre, n, post, has_active, act_type, true);
  } else {
    auto batch_arg =
        lite::kernels::host::GenBatchElementWiseArg<T>(x, y, param.Out, axis);
    X86CommonElementWise<T, int64_t, X86Config>::Run(batch_arg, op);
  }
  if (!elementwise_fn && !fast_bcast_fn) {
    LOG(FATAL) << "unsupported elementwise_compute called";
  }
}

#define ElementwiseOpCompute(op)                                              \
  template <typename T>                                                       \
  void Elementwise##op##Compute<T>::Run() {                                   \
    using X86Config = paddle::lite::x86::math::MergeConfig<                   \
        lite::x86::math::op##Config<T>,                                       \
        lite::x86::math::ActiveConfig<lite::x86::math::ActiveType::NO_ACTIVE, \
                                      T>>;                                    \
    elementwise_compute_template<operators::ElementwiseParam, T, X86Config>(  \
        this,                                                                 \
        lite::x86::math::Elementwise_Broadcast_##op<T>,                       \
        lite::x86::math::Elementwise_##op<T>,                                 \
        lite::x86::math::Naive##op<T>);                                       \
  }

#define ElementwiseOpActivationCompute(op)                                    \
  template <typename T>                                                       \
  void Elementwise##op##ActivationCompute<T>::Run() {                         \
    auto& param =                                                             \
        this->template Param<operators::FusionElementwiseActivationParam>();  \
    if (param.act_type == "relu") {                                           \
      using X86Config = paddle::lite::x86::math::MergeConfig<                 \
          lite::x86::math::op##Config<float>,                                 \
          lite::x86::math::ActiveConfig<lite::x86::math::ActiveType::RELU,    \
                                        float>>;                              \
      elementwise_compute_template<                                           \
          operators::FusionElementwiseActivationParam,                        \
          float,                                                              \
          X86Config>(this,                                                    \
                     lite::x86::math::Elementwise_Broadcast_##op<float>,      \
                     lite::x86::math::Elementwise_##op<float>,                \
                     lite::x86::math::Naive##op<float>,                       \
                     true,                                                    \
                     param.act_type);                                         \
    } else if (param.act_type == "tanh") {                                    \
      using X86Config = paddle::lite::x86::math::MergeConfig<                 \
          lite::x86::math::op##Config<float>,                                 \
          lite::x86::math::ActiveConfig<lite::x86::math::ActiveType::TANH,    \
                                        float>>;                              \
      elementwise_compute_template<                                           \
          operators::FusionElementwiseActivationParam,                        \
          float,                                                              \
          X86Config>(this,                                                    \
                     lite::x86::math::Elementwise_Broadcast_##op<float>,      \
                     lite::x86::math::Elementwise_##op<float>,                \
                     lite::x86::math::Naive##op<float>,                       \
                     true,                                                    \
                     param.act_type);                                         \
    } else if (param.act_type == "sigmoid") {                                 \
      using X86Config = paddle::lite::x86::math::MergeConfig<                 \
          lite::x86::math::op##Config<float>,                                 \
          lite::x86::math::ActiveConfig<lite::x86::math::ActiveType::SIGMOID, \
                                        float>>;                              \
      elementwise_compute_template<                                           \
          operators::FusionElementwiseActivationParam,                        \
          float,                                                              \
          X86Config>(this,                                                    \
                     lite::x86::math::Elementwise_Broadcast_##op<float>,      \
                     lite::x86::math::Elementwise_##op<float>,                \
                     lite::x86::math::Naive##op<float>,                       \
                     true,                                                    \
                     param.act_type);                                         \
    } else {                                                                  \
      LOG(FATAL) << "unsupported active type:" << param.act_type;             \
      using X86Config = paddle::lite::x86::math::MergeConfig<                 \
          lite::x86::math::op##Config<float>,                                 \
          lite::x86::math::                                                   \
              ActiveConfig<lite::x86::math::ActiveType::NO_ACTIVE, float>>;   \
      elementwise_compute_template<                                           \
          operators::FusionElementwiseActivationParam,                        \
          float,                                                              \
          X86Config>(this,                                                    \
                     lite::x86::math::Elementwise_Broadcast_##op<float>,      \
                     lite::x86::math::Elementwise_##op<float>,                \
                     lite::x86::math::Naive##op<float>,                       \
                     true,                                                    \
                     param.act_type);                                         \
    }                                                                         \
  }

// clang-format off
ElementwiseOpCompute(Add)
ElementwiseOpActivationCompute(Add)
ElementwiseOpCompute(Sub)
ElementwiseOpActivationCompute(Sub)
ElementwiseOpCompute(Mul)
ElementwiseOpActivationCompute(Mul)
ElementwiseOpCompute(Div)
ElementwiseOpActivationCompute(Div)
ElementwiseOpCompute(FloorDiv)
ElementwiseOpActivationCompute(FloorDiv)
ElementwiseOpCompute(Max)
ElementwiseOpActivationCompute(Max)
ElementwiseOpCompute(Min)
ElementwiseOpActivationCompute(Min)
ElementwiseOpCompute(Mod)
ElementwiseOpActivationCompute(Mod)
ElementwiseOpCompute(Pow)
ElementwiseOpActivationCompute(Pow)
// clang-format on

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseAddActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_sub_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseSubActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMulCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_mul_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseMulActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseDivCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_div_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseDivActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<int32_t>,
    int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_floordiv,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseFloorDivCompute<int64_t>,
    int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_pow,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwisePowCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_pow,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwisePowCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_pow,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwisePowCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseModCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseModCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMaxCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_max_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseMaxActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_min_activation,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::ElementwiseMinActivationCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ElementwiseMinCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();
