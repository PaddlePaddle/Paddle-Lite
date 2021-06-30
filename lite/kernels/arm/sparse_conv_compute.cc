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
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

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
  const float* din = input + first_ic * im_size;
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
  auto* din = input + first_ic * im_size;
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
  KERNEL_FUNC_NAME("sparse_conv_fp32_pipelined")
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
  auto* din = input + first_ic * im_size;
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
  KERNEL_FUNC_NAME("sparse_conv_fp32_pipelined")
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

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
    .BindInput("OcNonZeros", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Diffs", {LiteType::GetTensorTy(TARGET(kARM))})
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
