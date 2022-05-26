// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/sparse_conv_compute.h"
#include <utility>
#include "lite/backends/arm/math/sparse_conv_impl.h"
#include "lite/backends/arm/math/sparse_semi_conv_impl.h"
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
void SparseConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {}

template <>
void SparseConvCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.oc_nonzeros->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.oc_nonzeros->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  for (auto& ws : w_scale_) {
    ws *= input_scale;
  }
}

template <>
void SparseConvCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.oc_nonzeros->dims()[0]) {
    LOG(FATAL) << "weights scale size" << w_scale_.size()
               << "must equal to filter size" << param.oc_nonzeros->dims()[0];
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.oc_nonzeros->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  for (auto& ws : w_scale_) {
    ws = ws * input_scale / output_scale;
  }
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto* ptr = bias_.mutable_data<float>();
    auto* ptr_in = param.bias->data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / param.output_scale;
    }
    flag_trans_bias_ = true;
  }
  //! update relu6 parameter
  if (param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    param.activation_param.Relu_clipped_coef =
        param.activation_param.Relu_clipped_coef / param.output_scale;
  }
  //! update leaky_relu parameter
  if (param.activation_param.active_type ==
      lite_api::ActivationType::kLeakyRelu) {
    param.activation_param.Leaky_relu_alpha =
        param.activation_param.Leaky_relu_alpha / param.output_scale;
  }
  //! update hardswish parameter
  if (param.activation_param.active_type ==
      lite_api::ActivationType::kHardSwish) {
    param.activation_param.hard_swish_scale =
        param.activation_param.hard_swish_scale / param.output_scale;
    param.activation_param.hard_swish_offset =
        param.activation_param.hard_swish_offset / param.output_scale;
    param.activation_param.hard_swish_threshold =
        param.activation_param.hard_swish_threshold / param.output_scale;
  }
}

template <>
void SparseConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const float* input = param.x->data<float>();
  const float* nonzero_weights = param.nonzero_weights->data<float>();
  const int32_t* diffs = param.diffs->data<int32_t>();
  const uint32_t* oc_nonzeros = param.oc_nonzeros->data<uint32_t>();
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  float* dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int im_size = oh * ow;
  int first_ic = param.first_ic;
  int flag_semi = param.flag_semi;
  const float* din = input + first_ic * im_size;
  if (flag_semi == 1) {
    lite::arm::math::sparse_semi_conv_fp32_pipelined(nonzero_weights,
                                                     din,
                                                     diffs,
                                                     oc_nonzeros,
                                                     bias,
                                                     dout,
                                                     oc,
                                                     ic,
                                                     im_size,
                                                     param,
                                                     &ctx);
  } else {
    lite::arm::math::sparse_conv_fp32_pipelined(nonzero_weights,
                                                din,
                                                diffs,
                                                oc_nonzeros,
                                                bias,
                                                dout,
                                                oc,
                                                ic,
                                                im_size,
                                                param,
                                                &ctx);
  }
  KERNEL_FUNC_NAME("sparse_conv_fp32_pipelined")
}

template <>
void SparseConvCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto* input = param.x->data<int8_t>();
  auto* nonzero_weights = param.nonzero_weights->data<int8_t>();
  auto* diffs = param.diffs->data<int32_t>();
  auto* oc_nonzeros = param.oc_nonzeros->data<uint32_t>();
  auto* bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto* dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int im_size = oh * ow;
  int first_ic = param.first_ic;
  int flag_semi = param.flag_semi;
  auto* din = input + first_ic * im_size;
  if (flag_semi == 1) {
    lite::arm::math::sparse_semi_conv_int8_fp32_pipelined(nonzero_weights,
                                                          din,
                                                          diffs,
                                                          oc_nonzeros,
                                                          bias,
                                                          w_scale_.data(),
                                                          dout,
                                                          oc,
                                                          ic,
                                                          im_size,
                                                          param,
                                                          &ctx);
  } else {
    lite::arm::math::sparse_conv_int8_fp32_pipelined(nonzero_weights,
                                                     din,
                                                     diffs,
                                                     oc_nonzeros,
                                                     bias,
                                                     w_scale_.data(),
                                                     dout,
                                                     oc,
                                                     ic,
                                                     im_size,
                                                     param,
                                                     &ctx);
  }
  KERNEL_FUNC_NAME("sparse_conv_int8_fp32_pipelined")
}

template <>
void SparseConvCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto* input = param.x->data<int8_t>();
  auto* nonzero_weights = param.nonzero_weights->data<int8_t>();
  auto* diffs = param.diffs->data<int32_t>();
  auto* oc_nonzeros = param.oc_nonzeros->data<uint32_t>();
  auto* bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto* dout = param.output->mutable_data<int8_t>();

  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int im_size = oh * ow;
  int first_ic = param.first_ic;
  int flag_semi = param.flag_semi;
  auto* din = input + first_ic * im_size;
  if (flag_semi == 1) {
    lite::arm::math::sparse_semi_conv_int8_int8_pipelined(nonzero_weights,
                                                          din,
                                                          diffs,
                                                          oc_nonzeros,
                                                          bias,
                                                          w_scale_.data(),
                                                          dout,
                                                          oc,
                                                          ic,
                                                          im_size,
                                                          param,
                                                          &ctx);
  } else {
    lite::arm::math::sparse_conv_int8_int8_pipelined(nonzero_weights,
                                                     din,
                                                     diffs,
                                                     oc_nonzeros,
                                                     bias,
                                                     w_scale_.data(),
                                                     dout,
                                                     oc,
                                                     ic,
                                                     im_size,
                                                     param,
                                                     &ctx);
  }
  KERNEL_FUNC_NAME("sparse_conv_int8_int8_pipelined")
}

#ifdef ENABLE_ARM_FP16
template <>
void SparseConvCompute<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  // update weights
  auto& param = this->Param<param_t>();
  auto filter_tensor = param.nonzero_weights;
  if (filter_tensor->precision() == PRECISION(kFloat)) {
    Tensor tmp_tensor;
    tmp_tensor.CopyDataFrom(*filter_tensor);
    filter_tensor->clear();
    filter_tensor->set_precision(PRECISION(kFP16));
    float16_t* fp_data = filter_tensor->mutable_data<float16_t>();
    const float* in_data = tmp_tensor.data<float>();
    lite::arm::math::fp16::fp32_to_fp16(
        in_data, fp_data, filter_tensor->numel());
  }
  // update bias
  if (param.bias) {
    auto bias_tensor = param.bias;
    if (bias_tensor->precision() == PRECISION(kFloat)) {
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
  // update diffs
  auto diff_tensor = param.diffs;
  auto diff_data = diff_tensor->mutable_data<int32_t>();
  for (int i = 0; i < diff_tensor->numel(); i++) {
    diff_data[i] = diff_data[i] / 2;
  }
}

template <>
void SparseConvCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const float16_t* input = param.x->data<float16_t>();
  const float16_t* nonzero_weights = param.nonzero_weights->data<float16_t>();
  const int32_t* diffs = param.diffs->data<int32_t>();
  const uint32_t* oc_nonzeros = param.oc_nonzeros->data<uint32_t>();
  const float16_t* bias = param.bias ? param.bias->data<float16_t>() : nullptr;
  float16_t* dout = param.output->mutable_data<float16_t>();

  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int im_size = oh * ow;
  int first_ic = param.first_ic;
  int flag_semi = param.flag_semi;
  const float16_t* din = input + first_ic * im_size;
  if (flag_semi == 1) {
    lite::arm::math::fp16::sparse_semi_conv_fp16_pipelined(nonzero_weights,
                                                           din,
                                                           diffs,
                                                           oc_nonzeros,
                                                           bias,
                                                           dout,
                                                           oc,
                                                           ic,
                                                           im_size,
                                                           param,
                                                           &ctx);
  } else {
    lite::arm::math::fp16::sparse_conv_fp16_pipelined(nonzero_weights,
                                                      din,
                                                      diffs,
                                                      oc_nonzeros,
                                                      bias,
                                                      dout,
                                                      oc,
                                                      ic,
                                                      im_size,
                                                      param,
                                                      &ctx);
  }
  KERNEL_FUNC_NAME("sparse_conv_fp16_pipelined")
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::SparseConvCompute<PRECISION(kFP16),
                                                      PRECISION(kFP16)>
    SparseConvFp16;
REGISTER_LITE_KERNEL(sparse_conv2d, kARM, kFP16, kNCHW, SparseConvFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("NonZeroWeights",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("OcNonZeros",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Diffs",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm::SparseConvCompute<PRECISION(kFloat),
                                                      PRECISION(kFloat)>
    SparseConvFp32;
typedef paddle::lite::kernels::arm::SparseConvCompute<PRECISION(kInt8),
                                                      PRECISION(kFloat)>
    SparseConvInt8Fp32;
typedef paddle::lite::kernels::arm::SparseConvCompute<PRECISION(kInt8),
                                                      PRECISION(kInt8)>
    SparseConvInt8Int8;

REGISTER_LITE_KERNEL(sparse_conv2d, kARM, kFloat, kNCHW, SparseConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("NonZeroWeights", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OcNonZeros",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Diffs",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sparse_conv2d, kARM, kInt8, kNCHW, SparseConvInt8Fp32, int8_fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("NonZeroWeights",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("OcNonZeros",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Diffs",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sparse_conv2d, kARM, kInt8, kNCHW, SparseConvInt8Int8, int8_int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("NonZeroWeights",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("OcNonZeros",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Diffs",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();
