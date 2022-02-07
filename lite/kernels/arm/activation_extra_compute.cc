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

#include "lite/kernels/arm/activation_extra_compute.h"
#include "lite/backends/arm/math/funcs.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ReluClippedCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto coef = param.Relu_clipped_coef;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_clipped_relu<float>(
      x_data, output_data, x_dims.production(), coef, ctx.threads());
}

void SwishCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto beta = param.Swish_beta;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_swish<float>(
      x_data, output_data, x_dims.production(), beta, ctx.threads());
}

void LogCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_log<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void ExpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_exp<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void FloorCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_floor<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

template <>
void HardSigmoidCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float slope = param.hard_sigmoid_slope;
  float offset = param.hard_sigmoid_offset;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_hard_sigmoid<float>(
      x_data, output_data, x_dims.production(), slope, offset, ctx.threads());
}

#ifdef ENABLE_ARM_FP16
template <>
void HardSigmoidCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float16_t>();
  float slope = param.hard_sigmoid_slope;
  float offset = param.hard_sigmoid_offset;
  auto output_data = param.Out->mutable_data<float16_t>();
  lite::arm::math::fp16::act_hard_sigmoid<float16_t>(
      x_data, output_data, x_dims.production(), slope, offset, ctx.threads());
}

template <>
void HardSwishCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float16_t>();
  auto output_data = param.Out->mutable_data<float16_t>();
  float threshold = param.hard_swish_threshold;
  float scale = param.hard_swish_scale;
  float offset = param.hard_swish_offset;
  lite::arm::math::fp16::act_hard_swish<float16_t>(x_data,
                                                   output_data,
                                                   x_dims.production(),
                                                   threshold,
                                                   scale,
                                                   offset,
                                                   ctx.threads());
}
#endif

void SqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_sqrt<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void RsqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_rsqrt<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void SquareCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_square<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

template <>
void HardSwishCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float threshold = param.hard_swish_threshold;
  float scale = param.hard_swish_scale;
  float offset = param.hard_swish_offset;
  lite::arm::math::act_hard_swish<float>(x_data,
                                         output_data,
                                         x_dims.production(),
                                         threshold,
                                         scale,
                                         offset,
                                         ctx.threads());
}

void ReciprocalCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_reciprocal<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void AbsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_abs<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void GeluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  bool approximate = param.gelu_approximate;
  lite::arm::math::act_gelu<float>(
      x_data, output_data, x_dims.production(), approximate, ctx.threads());
}

template <typename T>
void ErfCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->template data<T>();
  auto output_data = param.Out->template mutable_data<T>();
  float alpha = param.Elu_alpha;
  lite::arm::math::erf<T>(
      x_data, output_data, x_dims.production(), ctx.threads());
}
template class ErfCompute<float>;

template <typename T>
void SignCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->template data<T>();
  auto output_data = param.Out->template mutable_data<T>();
  float alpha = param.Elu_alpha;
  lite::arm::math::sign<T>(
      x_data, output_data, x_dims.production(), ctx.threads());
}
template class SignCompute<float>;

template <typename T>
void SoftPlusCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->template data<T>();
  auto output_data = param.Out->template mutable_data<T>();
  float alpha = param.Elu_alpha;
  float beta = param.softplus_beta;
  lite::arm::math::softplus<T>(
      x_data, output_data, x_dims.production(), beta, ctx.threads());
}
template class SoftPlusCompute<float>;

template <typename T>
void MishCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->template data<T>();
  auto output_data = param.Out->template mutable_data<T>();
  float threshold = param.threshold;
  lite::arm::math::mish<T>(x_data, output_data, x_dims.production(), threshold);
}
template class MishCompute<float>;

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(
    hard_sigmoid,
    kARM,
    kFP16,
    kNCHW,
    paddle::lite::kernels::arm::HardSigmoidCompute<PRECISION(kFP16)>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    hard_swish,
    kARM,
    kFP16,
    kNCHW,
    paddle::lite::kernels::arm::HardSwishCompute<PRECISION(kFP16)>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(relu_clipped,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReluClippedCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Relu_clipped_coef", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    swish, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SwishCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("beta", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    log, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::LogCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    exp, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ExpCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    floor, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::FloorCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using fp32_hardsigmoid =
    paddle::lite::kernels::arm::HardSigmoidCompute<PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(hard_sigmoid, kARM, kFloat, kNCHW, fp32_hardsigmoid, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
REGISTER_LITE_KERNEL(
    sqrt, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    rsqrt, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::RsqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    square, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SquareCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    hard_swish,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::HardSwishCompute<PRECISION(kFloat)>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(reciprocal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReciprocalCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    abs, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::AbsCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gelu, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::GeluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using float_erf = paddle::lite::kernels::arm::ErfCompute<float>;
REGISTER_LITE_KERNEL(erf, kARM, kFloat, kNCHW, float_erf, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using float_sign = paddle::lite::kernels::arm::SignCompute<float>;
REGISTER_LITE_KERNEL(sign, kARM, kFloat, kNCHW, float_sign, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using float_softplus = paddle::lite::kernels::arm::SoftPlusCompute<float>;
REGISTER_LITE_KERNEL(softplus, kARM, kFloat, kNCHW, float_softplus, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using float_mish = paddle::lite::kernels::arm::MishCompute<float>;
REGISTER_LITE_KERNEL(mish, kARM, kFloat, kNCHW, float_mish, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
