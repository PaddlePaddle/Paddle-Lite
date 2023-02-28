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

#include "lite/kernels/arm/softmax_compute.h"
#include "lite/backends/arm/math/funcs.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
#ifdef LITE_WITH_ARM8_SVE2
#include "lite/backends/arm/math/sve/funcs_sve.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void SoftmaxCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<operators::SoftmaxParam>();
  const float* din = param.x->data<float>();
  float* dout = param.output->mutable_data<float>();
  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int axis_size = x_dims[axis];
  auto& ctx = this->ctx_->As<ARMContext>();
#ifdef LITE_WITH_ARM8_SVE2
  if (ctx.has_sve2()) {
    if (inner_num == 1) {
      lite::arm::math::sve::softmax_inner1_sve(din, dout, outer_num, axis_size);
    } else if (axis_size == 4) {
      lite::arm::math::sve::softmax_axis4_sve(
          din, dout, axis_size, inner_num, outer_num);
    } else {
      lite::arm::math::sve::softmax_basic_sve(
          din, dout, axis_size, inner_num, outer_num);
    }
    return;
  }
#endif

  if (inner_num == 1) {
    if (axis_size > 4) {
      lite::arm::math::softmax_inner1_large_axis(
          din, dout, outer_num, axis_size);
    } else {
      lite::arm::math::softmax_inner1_small_axis(
          din, dout, outer_num, axis_size);
    }
  } else {
    int compute_size = outer_num * inner_num;
    if (axis_size == 4 && inner_num % 8 == 0) {
      lite::arm::math::softmax_inner8_axis4(
          din, dout, axis_size, inner_num, outer_num);
    } else if (axis_size == 4 && inner_num % 4 == 0) {
      lite::arm::math::softmax_inner4_axis4(
          din, dout, axis_size, inner_num, outer_num);
    } else {
      if (inner_num % 8 == 0) {
        lite::arm::math::softmax_inner8(
            din, dout, axis_size, inner_num, outer_num);
      } else if (inner_num % 4 == 0) {
        lite::arm::math::softmax_inner4(
            din, dout, axis_size, inner_num, outer_num);
      } else {
        lite::arm::math::softmax_basic(
            din, dout, axis_size, inner_num, outer_num);
      }
    }
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void SoftmaxCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = Param<operators::SoftmaxParam>();
  const float16_t* din = param.x->data<float16_t>();
  float16_t* dout = param.output->mutable_data<float16_t>();
  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int axis_size = x_dims[axis];
  auto& ctx = this->ctx_->As<ARMContext>();
#ifdef LITE_WITH_ARM8_SVE2
  if (ctx.has_sve2()) {
    if (inner_num == 1) {
      lite::arm::math::sve::softmax_inner1_sve(din, dout, outer_num, axis_size);
    } else if (axis_size == 4) {
      lite::arm::math::sve::softmax_axis4_sve(
          din, dout, axis_size, inner_num, outer_num);
    } else {
      lite::arm::math::sve::softmax_basic_sve(
          din, dout, axis_size, inner_num, outer_num);
    }
    return;
  }
#endif

  if (inner_num == 1) {
    if (axis_size >= 8) {
      lite::arm::math::fp16::softmax_inner1_large_axis_fp16(
          din, dout, outer_num, axis_size);
    } else {
      lite::arm::math::fp16::softmax_inner1_small_axis_fp16(
          din, dout, outer_num, axis_size);
    }
  } else {
    int compute_size = outer_num * inner_num;
    if (axis_size == 4 && inner_num % 8 == 0) {
      lite::arm::math::fp16::softmax_inner8_axis4_fp16(
          din, dout, axis_size, inner_num, outer_num);
    } else {
      if (inner_num % 8 == 0) {
        lite::arm::math::fp16::softmax_inner8_axis1_fp16(
            din, dout, axis_size, inner_num, outer_num);
      } else {
        lite::arm::math::fp16::softmax_basic_fp16(
            din, dout, axis_size, inner_num, outer_num);
      }
    }
  }
}
#endif  // ENABLE_ARM_FP16
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::SoftmaxCompute<PRECISION(kFP16),
                                                   PRECISION(kFP16)>
    SoftmaxFp16;
REGISTER_LITE_KERNEL(softmax, kARM, kFP16, kNCHW, SoftmaxFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm::SoftmaxCompute<PRECISION(kFloat),
                                                   PRECISION(kFloat)>
    SoftmaxFp32;
REGISTER_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, SoftmaxFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
