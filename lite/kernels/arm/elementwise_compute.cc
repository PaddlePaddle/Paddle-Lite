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
#include <utility>
#include <vector>

#include "lite/backends/arm/math/elementwise_common_broadcast.h"
#include "lite/backends/arm/math/elementwise_common_broadcast_config.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/kernels/host/elementwise_op_func.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

namespace arm_math = paddle::lite::arm::math;

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

template <class T>
using FastBCastFn = void(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <class T>
using ElementWiseFn = void(const T* dinx, const T* diny, T* dout, int num);

template <class T>
using BinaryOpFn = lite::kernels::host::BinaryOpFn<T>;

enum class OprandSwapable { NO, YES };

template <class Elem_t, class DimValue_t, class NeonConfig>
struct CommonElementWiseOpArm {
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
          arm_math::neon_elementwise_range_to_one<NeonConfig>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      case (lite::kernels::host::BroadcastType::Y_AS_CONTINUOUS): {
        for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
          arm_math::neon_elementwise_one_to_range<NeonConfig>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      case (lite::kernels::host::BroadcastType::BOTH_CONTINUOUS): {
        for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
          arm_math::neon_elementwise_range_to_range<NeonConfig>(
              batch_arg.XAtBatch(batch_id),
              batch_arg.YAtBatch(batch_id),
              batch_arg.ZAtBatch(batch_id),
              range_length);
        }
        break;
      }
      default: {
        LOG(FATAL) << "Un supported bcast type";
        break;
      }
    }
  }
};

template <class Elem_t, class DimValue_t>
struct CommonElementWiseOpArm<Elem_t, DimValue_t, arm_math::NullNeonConfig> {
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
        LOG(FATAL) << "Un supported bcast type";
        break;
      }
    }
  }
};

// Note: All calling to elementwise_compute_template may set a template arg
//  "NeonConfig" to get better performance than the "NullNeonConfig".
//  However,it may increase the binary size significantly. So,
//  do it only if it is necessary.
//  see ElementwiseAddCompute<T, PType>::Run() to get an example.
template <class OpParamType,
          class T,
          OprandSwapable opd_swap_able,
          class NeonConfig>
void elementwise_compute_template(paddle::lite::KernelBase* kernel,
                                  FastBCastFn<T> fast_bcast_fn,
                                  ElementWiseFn<T> elementwise_fn,
                                  BinaryOpFn<T> op) {
  // Note:
  // Though this function will only access datas in operators::ElementwiseParam
  // the `kernel-> template Param<>()` requires to use true type from now
  // so `kernel->template Param<operators::ElementwiseParam>()` will not work
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
    elementwise_fn(x_data, y_data, out_data, x_dims.production());
  } else if (fast_bcast_fn &&
             is_fast_broadcast(x_dims, y_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(x_data, y_data, out_data, pre, n, post);
  } else if (fast_bcast_fn && opd_swap_able == OprandSwapable::YES &&
             axis == -1 &&
             is_fast_broadcast(y_dims, x_dims, axis, &pre, &n, &post)) {
    fast_bcast_fn(y_data, x_data, out_data, pre, n, post);
  } else if (elementwise_fn) {
    // todo: GenBatchElementWiseArg and common_elmentwise_op_arm can handle any
    //   kinds of "elementwise op", not only "broadcast". You could refactor the
    //   code to use only common_elmentwise_op_arm if necessary
    auto batch_arg =
        lite::kernels::host::GenBatchElementWiseArg<T>(x, y, param.Out, axis);
    // if NeonConfig is NullNeonConfig, a specialization that uses naive cpu
    // code will be called
    CommonElementWiseOpArm<T, int64_t, NeonConfig>::Run(batch_arg, op);
  }
  if (!elementwise_fn && !fast_bcast_fn) {
    LOG(FATAL) << "unsupported elementwise_compute called";
  }
}

template <typename T, PrecisionType PType>
void ElementwiseAddCompute<T, PType>::Run() {
  using NeonConfig = arm_math::MergeConfig<
      arm_math::AddConfig<T>,
      arm_math::ActiveConfig<arm_math::ActiveType::NO_ACTIVE, T>>;

  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::YES,
                               NeonConfig>(
      this,
      lite::arm::math::elementwise_add_broadcast<T>,
      lite::arm::math::elementwise_add<T>,
      paddle::lite::kernels::host::naive_add<T>);
}

template <typename T, PrecisionType PType>
void ElementwiseAddActivationCompute<T, PType>::Run() {
  // auto& param = Param<operators::FusionElementwiseActivationParam>();
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 T,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::elementwise_add_relu_broadcast<T>,
        lite::arm::math::elementwise_add_relu<T>,
        paddle::lite::kernels::host::naive_fused_op<
            T,
            paddle::lite::kernels::host::naive_add<T>,
            paddle::lite::kernels::host::naive_relu<T>>);
  }

  if (param.act_type == "tanh") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 T,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
        this,
        nullptr,
        lite::arm::math::elementwise_add_tanh<T>,
        paddle::lite::kernels::host::naive_fused_op<
            T,
            paddle::lite::kernels::host::naive_add<T>,
            paddle::lite::kernels::host::naive_tanh<T>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void ElementwiseAddCompute<float16_t, PRECISION(kFP16)>::Run() {
  using NeonConfig = arm_math::MergeConfig<
      arm_math::AddConfig<float16_t>,
      arm_math::ActiveConfig<arm_math::ActiveType::NO_ACTIVE, float16_t>>;

  elementwise_compute_template<operators::ElementwiseParam,
                               float16_t,
                               OprandSwapable::YES,
                               NeonConfig>(
      this,

      lite::arm::math::fp16::elementwise_add_broadcast<float16_t>,
      lite::arm::math::fp16::elementwise_add<float16_t>,
      paddle::lite::kernels::host::naive_add<float16_t>);
}

template <>
void ElementwiseAddActivationCompute<float16_t, PRECISION(kFP16)>::Run() {
  // auto& param = Param<operators::FusionElementwiseActivationParam>();
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float16_t,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::fp16::elementwise_add_relu_broadcast<float16_t>,
        lite::arm::math::fp16::elementwise_add_relu<float16_t>,
        paddle::lite::kernels::host::naive_fused_op<
            float16_t,
            paddle::lite::kernels::host::naive_add<float16_t>,
            paddle::lite::kernels::host::naive_relu<float16_t>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "fp16 unsupported Activation type: " << param.act_type;
  }
}

template <>
void ElementwiseMulCompute<float16_t, PRECISION(kFP16)>::Run() {
  using NeonConfig = arm_math::MergeConfig<
      arm_math::MulConfig<float16_t>,
      arm_math::ActiveConfig<arm_math::ActiveType::NO_ACTIVE, float16_t>>;

  elementwise_compute_template<operators::ElementwiseParam,
                               float16_t,
                               OprandSwapable::YES,
                               NeonConfig>(
      this,

      lite::arm::math::fp16::elementwise_mul_broadcast<float16_t>,
      lite::arm::math::fp16::elementwise_mul<float16_t>,
      paddle::lite::kernels::host::naive_mul<float16_t>);
}

template <>
void ElementwiseSubActivationCompute<float16_t, PRECISION(kFP16)>::Run() {
  // auto& param = Param<operators::FusionElementwiseActivationParam>();
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float16_t,
                                 OprandSwapable::NO,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::fp16::elementwise_sub_relu_broadcast<float16_t>,
        lite::arm::math::fp16::elementwise_sub_relu<float16_t>,
        paddle::lite::kernels::host::naive_fused_op<
            float16_t,
            paddle::lite::kernels::host::naive_sub<float16_t>,
            paddle::lite::kernels::host::naive_relu<float16_t>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "fp16 unsupported Activation type: " << param.act_type;
  }
}

template <>
void ElementwiseSubCompute<float16_t, PRECISION(kFP16)>::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               float16_t,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,

      lite::arm::math::fp16::elementwise_sub_broadcast<float16_t>,
      lite::arm::math::fp16::elementwise_sub<float16_t>,
      paddle::lite::kernels::host::naive_sub<float16_t>);
}

template <>
void ElementwiseDivActivationCompute<float16_t, PRECISION(kFP16)>::Run() {
// auto& param = Param<operators::FusionElementwiseActivationParam>();
#ifdef __aarch64__
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float16_t,
                                 OprandSwapable::NO,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::fp16::elementwise_div_relu_broadcast<float16_t>,
        lite::arm::math::fp16::elementwise_div_relu<float16_t>,
        paddle::lite::kernels::host::naive_fused_op<
            float16_t,
            paddle::lite::kernels::host::naive_div<float16_t>,
            paddle::lite::kernels::host::naive_relu<float16_t>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "fp16 unsupported Activation type: " << param.act_type;
  }
#else
  LOG(FATAL) << "it doesn't support v7 fp16 elementwise_div_act compute";
#endif
}

template <>
void ElementwiseDivCompute<float16_t, PRECISION(kFP16)>::Run() {
#ifdef __aarch64__
  elementwise_compute_template<operators::ElementwiseParam,
                               float16_t,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,

      lite::arm::math::fp16::elementwise_div_broadcast<float16_t>,
      lite::arm::math::fp16::elementwise_div<float16_t>,
      paddle::lite::kernels::host::naive_div<float16_t>);
#else
  LOG(FATAL) << "it doesn't support v7 fp16 elementwise_div compute";
#endif
}

template <>
void ElementwiseMulActivationCompute<float16_t, PRECISION(kFP16)>::Run() {
  // auto& param = Param<operators::FusionElementwiseActivationParam>();
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float16_t,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::fp16::elementwise_mul_relu_broadcast<float16_t>,
        lite::arm::math::fp16::elementwise_mul_relu<float16_t>,
        paddle::lite::kernels::host::naive_fused_op<
            float16_t,
            paddle::lite::kernels::host::naive_mul<float16_t>,
            paddle::lite::kernels::host::naive_relu<float16_t>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "fp16 unsupported Activation type: " << param.act_type;
  }
}
#endif

template <typename T, PrecisionType PType>
void ElementwiseSubCompute<T, PType>::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_sub_broadcast<T>,
      lite::arm::math::elementwise_sub<T>,
      paddle::lite::kernels::host::naive_sub<T>);
}

template <typename T, PrecisionType PType>
void ElementwiseSubActivationCompute<T, PType>::Run() {
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float,
                                 OprandSwapable::NO,
                                 arm_math::NullNeonConfig>(
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
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::YES,
                               arm_math::NullNeonConfig>(
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
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 T,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
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
  elementwise_compute_template<operators::ElementwiseParam,
                               float,
                               OprandSwapable::YES,
                               arm_math::NullNeonConfig>(
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
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
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

void ElementwiseMinCompute::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               float,
                               OprandSwapable::YES,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_min_broadcast<float>,
      lite::arm::math::elementwise_min<float>,
      paddle::lite::kernels::host::naive_min<float>);
}

void ElementwiseMinActivationCompute::Run() {
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float,
                                 OprandSwapable::YES,
                                 arm_math::NullNeonConfig>(
        this,
        lite::arm::math::elementwise_min_relu_broadcast<float>,
        lite::arm::math::elementwise_min_relu<float>,
        paddle::lite::kernels::host::naive_fused_op<
            float,
            paddle::lite::kernels::host::naive_min<float>,
            paddle::lite::kernels::host::naive_relu<float>>);
  }
  if (!act_supported) {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
}

template <typename T, PrecisionType PType>
void ElementwiseDivCompute<T, PType>::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_div_broadcast<T>,
      lite::arm::math::elementwise_div<T>,
      paddle::lite::kernels::host::naive_div<T>);
}

template <typename T, PrecisionType PType>
void ElementwiseFloorDivCompute<T, PType>::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_floor_div_broadcast<T>,
      lite::arm::math::elementwise_floor_div<T>,
      paddle::lite::kernels::host::naive_floor_div<T>);
}

template <typename T, PrecisionType PType>
void ElementwiseDivActivationCompute<T, PType>::Run() {
  auto& param =
      this->template Param<operators::FusionElementwiseActivationParam>();
  bool act_supported = false;
  if (param.act_type == "relu") {
    act_supported = true;
    elementwise_compute_template<operators::FusionElementwiseActivationParam,
                                 float,
                                 OprandSwapable::NO,
                                 arm_math::NullNeonConfig>(
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
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::NO,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_mod_broadcast<T>,
      lite::arm::math::elementwise_mod<T>,
      paddle::lite::kernels::host::naive_mod<T>);
}

template <typename T, PrecisionType PType>
void ElementwisePowCompute<T, PType>::Run() {
  elementwise_compute_template<operators::ElementwiseParam,
                               T,
                               OprandSwapable::YES,
                               arm_math::NullNeonConfig>(
      this,
      lite::arm::math::elementwise_pow_broadcast<T>,
      lite::arm::math::elementwise_pow<T>,
      paddle::lite::kernels::host::naive_pow<T>);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
using elementwise_add_fp16_t =
    paddle::lite::kernels::arm::ElementwiseAddCompute<float16_t,
                                                      PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kFP16, kNCHW, elementwise_add_fp16_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_add_fp16_t_act = paddle::lite::kernels::arm::
    ElementwiseAddActivationCompute<float16_t, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
                     kARM,
                     kFP16,
                     kNCHW,
                     elementwise_add_fp16_t_act,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_mul_fp16_t =
    paddle::lite::kernels::arm::ElementwiseMulCompute<float16_t,
                                                      PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kFP16, kNCHW, elementwise_mul_fp16_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_mul_fp16_t_act = paddle::lite::kernels::arm::
    ElementwiseMulActivationCompute<float16_t, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(fusion_elementwise_mul_activation,
                     kARM,
                     kFP16,
                     kNCHW,
                     elementwise_mul_fp16_t_act,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_sub_fp16_t =
    paddle::lite::kernels::arm::ElementwiseSubCompute<float16_t,
                                                      PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    elementwise_sub, kARM, kFP16, kNCHW, elementwise_sub_fp16_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_sub_fp16_t_act = paddle::lite::kernels::arm::
    ElementwiseSubActivationCompute<float16_t, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(fusion_elementwise_sub_activation,
                     kARM,
                     kFP16,
                     kNCHW,
                     elementwise_sub_fp16_t_act,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_div_fp16_t =
    paddle::lite::kernels::arm::ElementwiseDivCompute<float16_t,
                                                      PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    elementwise_div, kARM, kFP16, kNCHW, elementwise_div_fp16_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

using elementwise_div_fp16_t_act = paddle::lite::kernels::arm::
    ElementwiseDivActivationCompute<float16_t, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(fusion_elementwise_div_activation,
                     kARM,
                     kFP16,
                     kNCHW,
                     elementwise_div_fp16_t_act,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

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

using elementwise_add_int64_t =
    paddle::lite::kernels::arm::ElementwiseAddCompute<int64_t,
                                                      PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kInt64, kNCHW, elementwise_add_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
// float kernel has higher priority
using elementwise_add_int32_f =
    paddle::lite::kernels::arm::ElementwiseAddCompute<int32_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kFloat, kNCHW, elementwise_add_int32_f, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_add_int64_f =
    paddle::lite::kernels::arm::ElementwiseAddCompute<int64_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_add, kARM, kFloat, kNCHW, elementwise_add_int64_f, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA

using elementwise_add_float_t_act = paddle::lite::kernels::arm::
    ElementwiseAddActivationCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
                     kARM,
                     kFloat,
                     kNCHW,
                     elementwise_add_float_t_act,
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

#ifdef LITE_BUILD_EXTRA
// float kernel has higher priority
using elementwise_sub_int32_f =
    paddle::lite::kernels::arm::ElementwiseSubCompute<int32_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_sub, kARM, kFloat, kNCHW, elementwise_sub_int32_f, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_sub_int64_f =
    paddle::lite::kernels::arm::ElementwiseSubCompute<int64_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_sub, kARM, kFloat, kNCHW, elementwise_sub_int64_f, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA

using elementwise_sub_act_fp32 = paddle::lite::kernels::arm::
    ElementwiseSubActivationCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(fusion_elementwise_sub_activation,
                     kARM,
                     kFloat,
                     kNCHW,
                     elementwise_sub_act_fp32,
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
    paddle::lite::kernels::arm::ElementwiseMulCompute<int32_t,
                                                      PRECISION(kInt32)>;
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

#ifdef LITE_BUILD_EXTRA
// float kernel has higher priority
using elementwise_mul_int32_f =
    paddle::lite::kernels::arm::ElementwiseMulCompute<int32_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kFloat, kNCHW, elementwise_mul_int32_f, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_mul_int64_f =
    paddle::lite::kernels::arm::ElementwiseMulCompute<int64_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_mul, kARM, kFloat, kNCHW, elementwise_mul_int64_f, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA

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

REGISTER_LITE_KERNEL(elementwise_min,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ElementwiseMinCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_min_activation,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::ElementwiseMinActivationCompute,
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

using elementwise_div_int64_t =
    paddle::lite::kernels::arm::ElementwiseDivCompute<int64_t,
                                                      PRECISION(kInt64)>;

REGISTER_LITE_KERNEL(
    elementwise_div, kARM, kInt64, kNCHW, elementwise_div_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using elementwise_div_act_fp32 = paddle::lite::kernels::arm::
    ElementwiseDivActivationCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(fusion_elementwise_div_activation,
                     kARM,
                     kFloat,
                     kNCHW,
                     elementwise_div_act_fp32,
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

#ifdef LITE_BUILD_EXTRA
// float kernel has higher priority
using elementwise_mod_int64_f =
    paddle::lite::kernels::arm::ElementwiseModCompute<int64_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_mod, kARM, kFloat, kNCHW, elementwise_mod_int64_f, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using elementwise_mod_int32_f =
    paddle::lite::kernels::arm::ElementwiseModCompute<int32_t,
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    elementwise_mod, kARM, kFloat, kNCHW, elementwise_mod_int32_f, int32_mod)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA

using elementwise_pow_fp32_t =
    paddle::lite::kernels::arm::ElementwisePowCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    elementwise_pow, kARM, kFloat, kNCHW, elementwise_pow_fp32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using elementwise_pow_int32_t =
    paddle::lite::kernels::arm::ElementwisePowCompute<int32_t,
                                                      PRECISION(kInt32)>;

REGISTER_LITE_KERNEL(
    elementwise_pow, kARM, kInt32, kNCHW, elementwise_pow_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_floor_div_int32_t =
    paddle::lite::kernels::arm::ElementwiseFloorDivCompute<int32_t,
                                                           PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(elementwise_floordiv,
                     kARM,
                     kInt32,
                     kNCHW,
                     elementwise_floor_div_int32_t,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_floor_div_int64_t =
    paddle::lite::kernels::arm::ElementwiseFloorDivCompute<int64_t,
                                                           PRECISION(kInt64)>;

REGISTER_LITE_KERNEL(elementwise_floordiv,
                     kARM,
                     kInt64,
                     kNCHW,
                     elementwise_floor_div_int64_t,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
// float kernel has higher priority
using elementwise_floor_div_int32_f =
    paddle::lite::kernels::arm::ElementwiseFloorDivCompute<int32_t,
                                                           PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(elementwise_floordiv,
                     kARM,
                     kFloat,
                     kNCHW,
                     elementwise_floor_div_int32_f,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using elementwise_floor_div_int64_f =
    paddle::lite::kernels::arm::ElementwiseFloorDivCompute<int64_t,
                                                           PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(elementwise_floordiv,
                     kARM,
                     kFloat,
                     kNCHW,
                     elementwise_floor_div_int64_f,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
