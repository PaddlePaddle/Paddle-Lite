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

#include "lite/kernels/arm/fc_compute.h"
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

///  for fp32 kernel
template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

/// for int8 kernel with fp32 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->template Param<operators::FcParam>();
  /// update scale
  float input_scale = param.input_scale;
  int extend_size = flag_gemm_ ? m_ : n_;
  scale_.resize(extend_size);
  for (int i = 0; i < extend_size; ++i) {
    if (flag_gemm_) {
      scale_[i] = param.weight_scale[0] * input_scale;
    } else {
      scale_[i] = param.weight_scale[i] * input_scale;
    }
  }
}

/// for int8 kernel with int8 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->template Param<operators::FcParam>();
  /// update scale
  scale_ = param.weight_scale;
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  int extend_size = flag_gemm_ ? m_ : n_;
  scale_.resize(extend_size);
  for (int i = 0; i < extend_size; ++i) {
    if (flag_gemm_) {
      scale_[i] = param.weight_scale[0] * input_scale / output_scale;
    } else {
      scale_[i] = param.weight_scale[i] * input_scale / output_scale;
    }
  }
  /// update bias
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = bias_.data<float>();
    float out_scale = param.output_scale;
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / out_scale;
    }
    flag_trans_bias_ = true;
  }
}

template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto i_data = param.input->data<float>();
  auto o_data = param.output->mutable_data<float>();
  auto w_data = param.w->data<float>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  bool flag_act = false;
  lite_api::ActivationType act;
  if (param.activation_type == "relu") {
    act = lite_api::ActivationType::kRelu;
    flag_act = true;
  }
  if (flag_gemm_) {
    operators::ActivationParam act_param;
    act_param.has_active = false;
    lite::arm::math::sgemm(false,
                           false,
                           m_,
                           n_,
                           k_,
                           1.f,
                           i_data,
                           k_,
                           w_data,
                           n_,
                           0.f,
                           o_data,
                           n_,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
    if (param.bias) {
      CHECK_EQ(param.bias->numel(), n_);
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_, flag_act);
    }
  } else {
    for (int i = 0; i < m_; ++i) {
      auto i_data_batch = i_data + i * k_;
      auto o_data_batch = o_data + i * n_;
      lite::arm::math::sgemv(w_data,
                             i_data_batch,
                             o_data_batch,
                             false,
                             n_,
                             k_,
                             param.bias != nullptr,
                             b_data,
                             flag_act,
                             act,
                             &ctx);
    }
  }
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto i_data = param.input->data<int8_t>();
  auto o_data = param.output->mutable_data<float>();
  auto w_data = param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  bool flag_relu = false;
  operators::ActivationParam act_param;
  lite_api::ActivationType act;
  act_param.has_active = false;
  if (param.activation_type == "relu") {
    act = lite_api::ActivationType::kRelu;
    flag_relu = true;
  }
  if (flag_gemm_) {
    lite::arm::math::gemm_s8(false,
                             false,
                             m_,
                             n_,
                             k_,
                             i_data,
                             w_data,
                             o_data,
                             nullptr,
                             false,
                             scale_.data(),
                             act_param,
                             &ctx);
    if (param.bias) {
      CHECK_EQ(param.bias->numel(), n_);
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_, flag_relu);
    }
  } else {
    for (int i = 0; i < m_; ++i) {
      auto i_data_batch = i_data + i * k_;
      auto o_data_batch = o_data + i * n_;
      lite::arm::math::gemv_int8(w_data,
                                 i_data_batch,
                                 o_data_batch,
                                 false,
                                 n_,
                                 k_,
                                 scale_.data(),
                                 param.bias != nullptr,
                                 b_data,
                                 flag_relu,
                                 act,
                                 &ctx);
    }
  }
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto i_data = param.input->data<int8_t>();
  auto o_data = param.output->mutable_data<int8_t>();
  auto w_data = param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  bool flag_relu = false;
  operators::ActivationParam act_param;
  act_param.has_active = false;
  lite_api::ActivationType act;
  if (param.activation_type == "relu") {
    flag_relu = true;
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu;
    act = lite_api::ActivationType::kRelu;
  }
  if (flag_gemm_) {
    CHECK(!param.bias) << "fc int8 kernel with int8 output using gemm kernel "
                          "must not have bias";
    lite::arm::math::gemm_s8(false,
                             false,
                             m_,
                             n_,
                             k_,
                             i_data,
                             w_data,
                             o_data,
                             nullptr,
                             false,
                             scale_.data(),
                             act_param,
                             &ctx);
  } else {
    for (int i = 0; i < m_; ++i) {
      auto i_data_batch = i_data + i * k_;
      auto o_data_batch = o_data + i * n_;
      lite::arm::math::gemv_int8(w_data,
                                 i_data_batch,
                                 o_data_batch,
                                 false,
                                 n_,
                                 k_,
                                 scale_.data(),
                                 param.bias != nullptr,
                                 b_data,
                                 flag_relu,
                                 act,
                                 &ctx);
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::FcCompute<PRECISION(kFloat),
                                              PRECISION(kFloat)>
    FcCompute_FP32;
typedef paddle::lite::kernels::arm::FcCompute<PRECISION(kInt8),
                                              PRECISION(kFloat)>
    FcCompute_int8_fp32;
typedef paddle::lite::kernels::arm::FcCompute<PRECISION(kInt8),
                                              PRECISION(kInt8)>
    FcCompute_int8_int8;

REGISTER_LITE_KERNEL(fc, kARM, kFloat, kNCHW, FcCompute_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kARM, kInt8, kNCHW, FcCompute_int8_int8, int8out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kARM, kInt8, kNCHW, FcCompute_int8_fp32, fp32out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
