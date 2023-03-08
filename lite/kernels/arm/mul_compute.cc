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

#include "lite/kernels/arm/mul_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void MulCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

template <>
void MulCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.x->data<float>();
  const auto* y_data = param.y->data<float>();
  auto* o_data = param.output->mutable_data<float>();

  m_ = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  n_ = static_cast<int>(param.y->dims()
                            .Slice(param.y_num_col_dims, param.y->dims().size())
                            .production());

  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  k_ = x_w;
  auto& ctx = this->ctx_->template As<ARMContext>();
  operators::ActivationParam act_param;
  act_param.has_active = false;
  if (n_ == 1) {
    lite::arm::math::sgemv(x_data,
                           y_data,
                           o_data,
                           false,
                           m_,
                           k_,
                           0.f,
                           false,
                           nullptr,
                           act_param,
                           &ctx);

  } else {
    constexpr bool is_tranposed_y = false;
    int hblock = lite::arm::math::get_hblock(&ctx, m_);
    int m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(m_round * k_ * sizeof(float));

    float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                      ctx.llc_size() / sizeof(float);
    lite::arm::math::prepackA(
        packed_x, x_data, 1.f, k_, 0, m_, 0, k_, false, &ctx);
    int ldb = n_;
    if (is_tranposed_y) {
      ldb = k_;
    }
    lite::arm::math::sgemm_prepack(is_tranposed_y,
                                   m_,
                                   n_,
                                   k_,
                                   packed_x,
                                   y_data,
                                   ldb,
                                   0.f,
                                   o_data,
                                   n_,
                                   nullptr,
                                   false,
                                   act_param,
                                   &ctx);
  }
}

void mul_add_n_scale_bias(float* o_data, float* scale_, int m_, int n_) {
  float32x4_t bias_v, scale_v, out_v, tmp_v;
  int n_tail = n_ % 4;
  int n_inner = n_ - n_tail;
  for (int i = 0; i < m_; i++) {
    for (int j = 0; j < n_inner; j += 4) {
      tmp_v = vld1q_f32(&o_data[i * n_ + j]);
      scale_v = vld1q_f32(&scale_[j]);
      out_v = vmulq_f32(scale_v, tmp_v);
      vst1q_f32(&o_data[i * n_ + j], out_v);
    }
    for (int j = n_inner; j < n_; j++) {
      o_data[i * n_ + j] *= scale_[j];
    }
  }
}

template <>
void MulCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

template <>
void MulCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.x->data<int8_t>();
  const auto* y_data = param.y->data<int8_t>();
  auto* o_data = param.output->mutable_data<float>();

  m_ = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  n_ = static_cast<int>(param.y->dims()
                            .Slice(param.y_num_col_dims, param.y->dims().size())
                            .production());

  scale_.resize(n_);
  scale_one.resize(m_);
  if (param.weight_scale.size() == 1) {
    param.output_scale = param.input_scale * param.weight_scale[0];
    for (int i = 0; i < n_; i++) {
      scale_[i] = param.output_scale;
    }
  } else {
    for (int i = 0; i < n_; i++) {
      param.output_scale = param.input_scale * param.weight_scale[i];
      scale_[i] = param.output_scale;
    }
  }
  for (int i = 0; i < m_; i++) {
    scale_one[i] = 1;
  }

  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  k_ = x_w;
  auto& ctx = this->ctx_->template As<ARMContext>();
  operators::ActivationParam act_param;
  act_param.has_active = false;
  if (n_ == 1) {
    lite::arm::math::gemv_int8(x_data,
                               y_data,
                               o_data,
                               false,
                               m_,
                               k_,
                               scale_one.data(),
                               false,
                               nullptr,
                               act_param,
                               &ctx);
  } else {
    constexpr bool is_tranposed_y = false;
    int ldb = n_;
    if (is_tranposed_y) {
      ldb = k_;
    }
    lite::arm::math::gemm_s8(is_tranposed_y,
                             false,
                             false,
                             m_,
                             n_,
                             k_,
                             x_data,
                             y_data,
                             o_data,
                             nullptr,
                             false,
                             lite::arm::math::GemmNoBias,
                             scale_one.data(),
                             act_param,
                             &ctx);
  }
  mul_add_n_scale_bias(o_data, scale_.data(), m_, n_);
}
#ifdef ENABLE_ARM_FP16
template <>
void MulCompute<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

template <>
void MulCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.x->data<float16_t>();
  const auto* y_data = param.y->data<float16_t>();
  auto* o_data = param.output->mutable_data<float16_t>();

  m_ = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  n_ = static_cast<int>(param.y->dims()
                            .Slice(param.y_num_col_dims, param.y->dims().size())
                            .production());

  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  k_ = x_w;
  auto& ctx = this->ctx_->template As<ARMContext>();
  operators::ActivationParam act_param;
  act_param.has_active = false;
  if (n_ == 1) {
    lite::arm::math::fp16::gemv_fp16(x_data,
                                     y_data,
                                     o_data,
                                     false,
                                     m_,
                                     k_,
                                     0.f,
                                     false,
                                     nullptr,
                                     act_param.has_active,
                                     act_param,
                                     &ctx);

  } else {
    constexpr bool is_tranposed_y = false;
    int hblock = lite::arm::math::get_hblock(&ctx, m_);
    int m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(m_round * k_ * sizeof(float16_t));

    float16_t* packed_x =
        static_cast<float16_t*>(ctx.workspace_data<float16_t>()) +
        ctx.llc_size() / sizeof(float16_t);
    lite::arm::math::fp16::prepackA_fp16(
        packed_x, x_data, 1.f, k_, 0, m_, 0, k_, false, &ctx);
    int ldb = n_;
    if (is_tranposed_y) {
      ldb = k_;
    }
    lite::arm::math::fp16::gemm_prepack_fp16(is_tranposed_y,
                                             m_,
                                             n_,
                                             k_,
                                             packed_x,
                                             y_data,
                                             ldb,
                                             0.f,
                                             o_data,
                                             n_,
                                             nullptr,
                                             false,
                                             act_param,
                                             &ctx);
  }
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::MulCompute<PRECISION(kFloat),
                                               PRECISION(kFloat)>
    Mul_f32_f32;

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::MulCompute<PRECISION(kFP16),
                                               PRECISION(kFP16)>
    Mul_f16_f16;
REGISTER_LITE_KERNEL(mul, kARM, kFP16, kNCHW, Mul_f16_f16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(mul, kARM, kFloat, kNCHW, Mul_f32_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

typedef paddle::lite::kernels::arm::MulCompute<PRECISION(kInt8),
                                               PRECISION(kFloat)>
    Mul_int8_f32;

REGISTER_LITE_KERNEL(mul, kARM, kInt8, kNCHW, Mul_int8_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
