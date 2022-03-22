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

#include "lite/kernels/arm/activation_compute.h"
#include "lite/backends/arm/math/funcs.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <>
void ReluCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_relu<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

#ifdef ENABLE_ARM_FP16
template <>
void ReluCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float16_t>();
  auto output_data = param.Out->mutable_data<float16_t>();
  lite::arm::math::fp16::act_relu<float16_t>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

template <>
void PReluCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float16_t>();
  auto mode = param.Prelu_mode;
  auto alpha_data = param.Prelu_alpha->data<float16_t>();
  auto output_data = param.Out->mutable_data<float16_t>();

  int outer_size = x_dims[0];
  int channel_size = x_dims[1];
  int inner_size = x_dims.count(2, x_dims.size());

  lite::arm::math::fp16::act_prelu<float16_t>(x_data,
                                              output_data,
                                              outer_size,
                                              channel_size,
                                              inner_size,
                                              mode,
                                              alpha_data,
                                              ctx.threads());
}
template <>
void TanhCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float16_t>();
  auto output_data = param.Out->mutable_data<float16_t>();
  lite::arm::math::fp16::act_tanh<float16_t>(
      x_data, output_data, x_dims.production(), ctx.threads());
}
#endif

void LeakyReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto alpha = param.Leaky_relu_alpha;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_relu_neg<float>(
      x_data, output_data, x_dims.production(), alpha, ctx.threads());
}

template <>
void PReluCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto mode = param.Prelu_mode;
  auto alpha_data = param.Prelu_alpha->data<float>();
  auto output_data = param.Out->mutable_data<float>();

  int outer_size = x_dims[0];
  int channel_size = x_dims[1];
  int inner_size = x_dims.count(2, x_dims.size());

  lite::arm::math::act_prelu<float>(x_data,
                                    output_data,
                                    outer_size,
                                    channel_size,
                                    inner_size,
                                    mode,
                                    alpha_data,
                                    ctx.threads());
}

void SigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_sigmoid<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

template <>
void TanhCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_tanh<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void Relu6Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float coef = 6.;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_clipped_relu<float>(
      x_data, output_data, x_dims.production(), coef, ctx.threads());
}

void ThresholdedReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float threshold = param.relu_threshold;
  lite::arm::math::act_thresholded_relu<float>(
      x_data, output_data, x_dims.production(), threshold, ctx.threads());
}

void EluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float alpha = param.Elu_alpha;
  lite::arm::math::act_elu<float>(
      x_data, output_data, x_dims.production(), alpha, ctx.threads());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(relu,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::ReluCompute<PRECISION(kFP16)>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(prelu,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::PReluCompute<PRECISION(kFP16)>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(tanh,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::TanhCompute<PRECISION(kFP16)>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(relu,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReluCompute<PRECISION(kFloat)>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(leaky_relu,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LeakyReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("leaky_relu", 1)
    .Finalize();
REGISTER_LITE_KERNEL(
    prelu,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::PReluCompute<PRECISION(kFloat)>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("mode", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(sigmoid,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(tanh,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::TanhCompute<PRECISION(kFloat)>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    relu6, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::Relu6Compute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(thresholded_relu,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ThresholdedReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    elu, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::EluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
