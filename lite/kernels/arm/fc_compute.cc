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
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void naive_transpose(const Dtype* din, Dtype* dout, int m, int n) {
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      dout[k++] = din[j * n + i];
    }
  }
}

template <PrecisionType PType>
void fc_trans_weights(const Tensor& tin, Tensor* tout);

template <>
void fc_trans_weights<PRECISION(kFloat)>(const Tensor& tin, Tensor* tout) {
  CHECK_EQ(tin.dims().size(), 2) << "fc weights size must = 2";
  int m = tin.dims()[0];
  int n = tin.dims()[1];
  tout->Resize({n, m});
  auto* ptr_in = tin.data<float>();
  auto* ptr_out = tout->mutable_data<float>();
  naive_transpose(ptr_in, ptr_out, m, n);
}

template <>
void fc_trans_weights<PRECISION(kInt8)>(const Tensor& tin, Tensor* tout) {
  CHECK_EQ(tin.dims().size(), 2) << "fc weights size must = 2";
  int m = tin.dims()[0];
  int n = tin.dims()[1];
  tout->Resize({n, m});
  auto* ptr_in = tin.data<int8_t>();
  auto* ptr_out = tout->mutable_data<int8_t>();
  naive_transpose(ptr_in, ptr_out, m, n);
}

template <PrecisionType PType, PrecisionType OutType>
bool check_fc_use_gemm(int m, const std::vector<float>& scale, bool has_bias) {
  return m > 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kFloat)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK_GT(scale.size(), 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kInt8)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK_GT(scale.size(), 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1 && !has_bias;
}

template <PrecisionType PType, PrecisionType OutType>
void FcCompute<PType, OutType>::ReInitWhenNeeded() {
  auto& param = this->template Param<operators::FcParam>();
  auto x_dims = param.input->dims();
  if (last_shape_ == x_dims) {
    return;
  }
  last_shape_ = x_dims;
  auto w_dims = param.w->dims();
  auto& ctx = this->ctx_->template As<ARMContext>();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);
  CHECK_GE(param.output->dims().size(), 2UL);
  int in_num_col_dims = param.in_num_col_dims;
  std::string op_type = param.op_type;
  if (op_type == "matmul" || op_type == "matmul_v2") {
    in_num_col_dims = x_dims.size() - 1;
  }

  m_ = x_dims.Slice(0, in_num_col_dims).production();
  k_ = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k_, w_dims[0]);
  n_ = w_dims[1];
  flag_gemm_ = check_fc_use_gemm<PType, OutType>(
      m_, param.weight_scale, param.bias != nullptr);
  if (!flag_trans_weights_ && !flag_gemm_) {
    flag_trans_weights_ = true;
    fc_trans_weights<PType>(*param.w, &weights_);
  }
}

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
    auto* ptr = bias_.mutable_data<float>();
    auto* ptr_in = param.bias->data<float>();
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

  auto* i_data = param.input->data<float>();
  auto* o_data = param.output->mutable_data<float>();
  auto* w_data = flag_gemm_ ? param.w->data<float>() : weights_.data<float>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  operators::ActivationParam act_param;
  act_param.has_active = false;
  if (flag_gemm_) {
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
      if (param.activation_type == "relu") {
        act_param.has_active = true;
        act_param.active_type = lite_api::ActivationType::kRelu;
      } else if (param.activation_type == "relu6") {
        act_param.has_active = true;
        act_param.active_type = lite_api::ActivationType::kRelu6;
        act_param.Relu_clipped_coef = param.alpha;
      }
      CHECK_EQ(param.bias->numel(), n_);
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_, &act_param);
    }
  } else {
    if (param.activation_type == "relu") {
      act_param.has_active = true;
      act_param.active_type = lite_api::ActivationType::kRelu;
    } else if (param.activation_type == "relu6") {
      act_param.has_active = true;
      act_param.active_type = lite_api::ActivationType::kRelu6;
      act_param.Relu_clipped_coef = param.alpha;
    }
    for (int i = 0; i < m_; ++i) {
      auto* i_data_batch = i_data + i * k_;
      auto* o_data_batch = o_data + i * n_;
      lite::arm::math::sgemv(w_data,
                             i_data_batch,
                             o_data_batch,
                             false,
                             n_,
                             k_,
                             0.f,
                             param.bias != nullptr,
                             b_data,
                             act_param,
                             &ctx);
    }
  }
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto* i_data = param.input->data<int8_t>();
  auto* o_data = param.output->mutable_data<float>();
  auto* w_data =
      flag_trans_weights_ ? weights_.data<int8_t>() : param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  operators::ActivationParam act_param;
  // lite_api::ActivationType act;
  act_param.has_active = false;
  if (param.activation_type == "relu") {
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu;
  } else if (param.activation_type == "relu6") {
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu6;
    act_param.Relu_clipped_coef = param.alpha;
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
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_, &act_param);
    }
  } else {
    for (int i = 0; i < m_; ++i) {
      auto* i_data_batch = i_data + i * k_;
      auto* o_data_batch = o_data + i * n_;
      lite::arm::math::gemv_int8(w_data,
                                 i_data_batch,
                                 o_data_batch,
                                 false,
                                 n_,
                                 k_,
                                 scale_.data(),
                                 param.bias != nullptr,
                                 b_data,
                                 act_param,
                                 &ctx);
    }
  }
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto* i_data = param.input->data<int8_t>();
  auto* o_data = param.output->mutable_data<int8_t>();
  auto* w_data =
      flag_trans_weights_ ? weights_.data<int8_t>() : param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  operators::ActivationParam act_param;
  act_param.has_active = false;
  if (param.activation_type == "relu") {
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu;
  } else if (param.activation_type == "relu6") {
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu6;
    act_param.Relu_clipped_coef = param.alpha;
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
      auto* i_data_batch = i_data + i * k_;
      auto* o_data_batch = o_data + i * n_;
      lite::arm::math::gemv_int8(w_data,
                                 i_data_batch,
                                 o_data_batch,
                                 false,
                                 n_,
                                 k_,
                                 scale_.data(),
                                 param.bias != nullptr,
                                 b_data,
                                 act_param,
                                 &ctx);
    }
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void fc_trans_weights<PRECISION(kFP16)>(const Tensor& tin, Tensor* tout) {
  CHECK_EQ(tin.dims().size(), 2) << "fc weights size must = 2";
  int m = tin.dims()[0];
  int n = tin.dims()[1];
  tout->Resize({n, m});
  auto* ptr_in = tin.data<float16_t>();
  auto* ptr_out = tout->mutable_data<float16_t>();
  naive_transpose(ptr_in, ptr_out, m, n);
}

template <>
void FcCompute<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  auto& param1 = this->template Param<operators::FcParam>();
  auto filter_tensor = param1.w;
  if (filter_tensor->precision() != PRECISION(kFP16)) {
    Tensor tmp_tensor;
    tmp_tensor.CopyDataFrom(*filter_tensor);
    filter_tensor->clear();
    filter_tensor->set_precision(PRECISION(kFP16));
    float16_t* fp_data = filter_tensor->mutable_data<float16_t>();
    const float* in_data = tmp_tensor.data<float>();
    lite::arm::math::fp16::fp32_to_fp16(
        in_data, fp_data, filter_tensor->numel());
  }
  if (param1.bias) {
    auto bias_tensor = param1.bias;
    if (bias_tensor->precision() != PRECISION(kFP16)) {
      Tensor tmp_tensor;
      tmp_tensor.CopyDataFrom(*bias_tensor);
      bias_tensor->clear();
      bias_tensor->set_precision(PRECISION(kFP16));
      float16_t* fp_data = bias_tensor->mutable_data<float16_t>();
      const float* in_data = tmp_tensor.data<float>();
      lite::arm::math::fp16::fp32_to_fp16(
          in_data, fp_data, bias_tensor->numel());
    }
  }
  ReInitWhenNeeded();
}

template <>
void FcCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto* i_data = param.input->data<float16_t>();
  auto* o_data = param.output->mutable_data<float16_t>();
  auto* w_data =
      flag_gemm_ ? param.w->data<float16_t>() : weights_.data<float16_t>();
  const float16_t* b_data =
      param.bias ? param.bias->data<float16_t>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float16_t>();
  }
  operators::ActivationParam act_param;
  if (flag_gemm_) {
    act_param.has_active = false;
    lite::arm::math::fp16::sgemm_fp16(false,
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
      if (param.activation_type == "relu") {
        act_param.has_active = true;
        act_param.active_type = lite_api::ActivationType::kRelu;
      } else if (param.activation_type == "relu6") {
        act_param.has_active = true;
        act_param.active_type = lite_api::ActivationType::kRelu6;
        act_param.Relu_clipped_coef = param.alpha;
      }
      lite::arm::math::fp16::fill_bias_fc(o_data, b_data, m_, n_, &act_param);
    }
  } else {
    if (param.activation_type == "relu") {
      act_param.has_active = true;
      act_param.active_type = lite_api::ActivationType::kRelu;
    } else if (param.activation_type == "relu6") {
      act_param.has_active = true;
      act_param.active_type = lite_api::ActivationType::kRelu6;
      act_param.Relu_clipped_coef = param.alpha;
    }
    for (int i = 0; i < m_; ++i) {
      auto* i_data_batch = i_data + i * k_;
      auto* o_data_batch = o_data + i * n_;
      lite::arm::math::fp16::gemv_fp16(w_data,
                                       i_data_batch,
                                       o_data_batch,
                                       false,
                                       n_,
                                       k_,
                                       0.f,
                                       param.bias != nullptr,
                                       b_data,
                                       act_param.has_active,
                                       act_param,
                                       &ctx);
    }
  }
}

#endif
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

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::FcCompute<PRECISION(kFP16),
                                              PRECISION(kFP16)>
    FcCompute_FP16;
REGISTER_LITE_KERNEL(fc, kARM, kFP16, kNCHW, FcCompute_FP16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(fc, kARM, kFloat, kNCHW, FcCompute_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
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
