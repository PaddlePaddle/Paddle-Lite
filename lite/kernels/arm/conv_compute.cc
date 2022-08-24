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

#include "lite/kernels/arm/conv_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/arm/conv_depthwise.h"
#include "lite/kernels/arm/conv_depthwise_common.h"
#include "lite/kernels/arm/conv_direct.h"
#include "lite/kernels/arm/conv_gemmlike.h"
#include "lite/kernels/arm/conv_winograd.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
#define PARAM_INIT                                                           \
  auto& param = this->Param<param_t>();                                      \
  auto w_dims = param.filter->dims();                                        \
  auto& ctx = this->ctx_->template As<ARMContext>();                         \
  auto paddings = *param.paddings;                                           \
  auto dilations = *param.dilations;                                         \
  int ic = w_dims[1] * param.groups;                                         \
  int oc = w_dims[0];                                                        \
  int kh = w_dims[2];                                                        \
  int kw = w_dims[3];                                                        \
  int pad_h = paddings[0];                                                   \
  int pad_w = paddings[2];                                                   \
  int stride = param.strides[0];                                             \
  int sh = param.strides[1];                                                 \
  int sw = param.strides[0];                                                 \
  int threads = ctx.threads();                                               \
  int chin = param.x->dims()[1];                                             \
  int hin = param.x->dims()[2];                                              \
  int win = param.x->dims()[3];                                              \
  int chout = param.output->dims()[1];                                       \
  int hout = param.output->dims()[2];                                        \
  int wout = param.output->dims()[3];                                        \
  bool pads_equal =                                                          \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));        \
  bool pads_all_equal = (pads_equal && pad_h == pad_w);                      \
  bool ks_equal = (sw == sh) && (kw == kh);                                  \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);             \
  bool kps_equal = (pad_h == pad_w) && ks_equal;                             \
  bool flag_dw_3x3 = (kw == 3) && (kh == 3) && (stride == 1 || stride == 2); \
  bool flag_dw_5x5 = (kw == 5) && (kh == 5) && (stride == 1 || stride == 2); \
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

template <>
void ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PARAM_INIT
  /// select conv impl
  if (param.groups == ic && ic == oc && ks_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && ks_equal &&
             no_dilation) {
    impl_ = new WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking winograd conv";
  } else if (param.groups == 1 && kw == 3 && stride == 2 &&
             chin * chout < 4 * hin * win && ks_equal && no_dilation) {
    impl_ = new DirectConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking direct conv";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking gemm like conv";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

template <>
void ConvCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  PARAM_INIT
  if (param.groups == ic && ic == oc && kps_equal && pads_equal &&
      no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run DepthwiseConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 2 && sh == 2 &&
             no_dilation && pads_equal && ks_equal &&
             (!ctx.has_dot() || ctx.has_sve2())) {
    impl_ = new DirectConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run DirectConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 1 && no_dilation &&
             pads_equal && ks_equal && !ctx.has_dot()) {
    impl_ = new WinogradConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run WinogradConv Int8";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run GemmLikeConvInt8";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

template <>
void ConvCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  PARAM_INIT
  if (param.groups == ic && ic == oc && kps_equal && pads_equal &&
      no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run DepthwiseConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 2 && sh == 2 &&
             no_dilation && pads_equal && ks_equal &&
             (!ctx.has_dot() || ctx.has_sve2())) {
    impl_ = new DirectConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run DirectConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 1 && no_dilation &&
             pads_equal && ks_equal && !ctx.has_dot()) {
    impl_ = new WinogradConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run WinogradConv Int8";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run GemmLikeConvInt8";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

#ifdef ENABLE_ARM_FP16
template <>
void ConvCompute<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  PARAM_INIT
  /// select conv impl
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  bool has_active = act_param.has_active;
  bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
  bool conv_3x3_wino = (ic <= 8) || (oc <= 8);
  bool stride_less = (sw == 1) || (sw == 2);
  if (param.groups == ic && ic == oc) {
    if (no_dilation && stride_less &&
        ((flag_dw_5x5 && ks_equal) ||
         (flag_dw_3x3 && kps_equal && pads_less))) {
      impl_ = new DepthwiseConv<PRECISION(kFP16), PRECISION(kFP16)>;
    } else {
      impl_ = new DepthwiseConvCommon<PRECISION(kFP16), PRECISION(kFP16)>;
    }
  } else if (param.groups == 1 && kw == 3 && sw == 2 && no_dilation &&
             chin * chout < 4 * hin * win && ks_equal) {
    impl_ = new DirectConv<PRECISION(kFP16), PRECISION(kFP16)>;
  } else if (param.groups == 1 && kw == 3 && sw == 1 && no_dilation &&
             ks_equal) {
    if (conv_3x3_wino) {
      impl_ = new DirectConv<PRECISION(kFP16), PRECISION(kFP16)>;
    } else {
      impl_ = new WinogradConv<PRECISION(kFP16), PRECISION(kFP16)>;
    }
  } else {
    impl_ = new GemmLikeConv<PRECISION(kFP16), PRECISION(kFP16)>;
  }
  // when running op python unit_test, the weight dtype is float
  auto filter_tensor = param.filter;
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
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}
#endif
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::ConvCompute<PRECISION(kFloat),
                                                PRECISION(kFloat)>
    ConvFp32;
typedef paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                                PRECISION(kFloat)>
    ConvInt8_Fp32;
typedef paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                                PRECISION(kInt8)>
    ConvInt8_Int8;

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::ConvCompute<PRECISION(kFP16),
                                                PRECISION(kFP16)>
    ConvFp16;

REGISTER_LITE_KERNEL(conv2d, kARM, kFP16, kNCHW, ConvFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kARM, kFP16, kNCHW, ConvFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("SecondInput", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Prelu_alpha",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Prelu_alpha",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Prelu_alpha",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Prelu_alpha",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
