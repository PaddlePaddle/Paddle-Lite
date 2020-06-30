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
#include "lite/kernels/arm/conv_direct.h"
#include "lite/kernels/arm/conv_gemmlike.h"
#include "lite/kernels/arm/conv_winograd.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto w_dims = param.filter->dims();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int ic = w_dims[1] * param.groups;
  int oc = w_dims[0];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int pad = paddings[0];
  int stride = param.strides[0];
  int threads = ctx.threads();

  int chin = param.x->dims()[1];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];
  int chout = param.output->dims()[1];
  int hout = param.output->dims()[2];
  int wout = param.output->dims()[3];

  bool pads_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
  bool pads_all_equal = (pads_equal && paddings[0] == paddings[2]);

  bool ks_equal = (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);

  bool flag_dw_3x3 = (kw == 3) && (kh == 3) && (stride == 1 || stride == 2);
  bool flag_dw_5x5 = (kw == 5) && (kh == 5) && (stride == 1 || stride == 2);

#ifdef __aarch64__
#else
  bool flag =
      (stride == 1 && (paddings[0] > 1 || paddings[2] > 1)) ? false : true;
  flag_dw_3x3 = flag_dw_3x3 && flag;
#endif
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  /// select conv impl
  if (param.groups == ic && ic == oc && ks_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && ks_equal &&
             no_dilation) {
    // TODO(MyPandaShaoxiang): winograd conv support any pad
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
  auto& param = this->Param<param_t>();
  auto w_dims = param.filter->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  bool pads_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
  int ic = param.groups * w_dims[1];
  int oc = w_dims[0];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int ph = paddings[0];
  int pw = paddings[2];
  int sh = param.strides[1];
  int sw = param.strides[0];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];
  bool pads_all_equal = (pads_equal && paddings[0] == paddings[2]);

  bool kps_equal = (pw == ph) && (sh == sw) && (kw == kh);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool flag_dw_3x3 = (kw == 3 && kh == 3 && (sw == 1 || sw == 2));
  bool flag_dw_5x5 = pads_all_equal && (kw == 5 && (sw == 1 || sw == 2));
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
  if (param.groups == ic && ic == oc && kps_equal && pads_equal &&
      no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run DepthwiseConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 2 && no_dilation &&
             pads_equal) {
    impl_ = new DirectConv<PRECISION(kInt8), PRECISION(kFloat)>;
    // VLOG(3) << "Run DirectConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 1 && no_dilation &&
             pads_equal) {
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
  auto& param = this->Param<param_t>();
  auto w_dims = param.filter->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  bool pads_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

  int ic = w_dims[1] * param.groups;
  int oc = w_dims[0];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int ph = paddings[0];
  int pw = paddings[2];
  int sh = param.strides[1];
  int sw = param.strides[0];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];
  bool pads_all_equal = (pads_equal && paddings[0] == paddings[2]);

  bool kps_equal = (pw == ph) && (sh == sw) && (kw == kh);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool flag_dw_3x3 = (kw == 3 && kh == 3 && (sw == 1 || sw == 2));
  bool flag_dw_5x5 = pads_all_equal && (kw == 5 && (sw == 1 || sw == 2));
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  if (param.groups == ic && ic == oc && kps_equal && pads_equal &&
      no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run DepthwiseConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 2 && no_dilation &&
             pads_equal) {
    impl_ = new DirectConv<PRECISION(kInt8), PRECISION(kInt8)>;
    // VLOG(3) << "Run DirectConv Int8";
  } else if (param.groups == 1 && kw == 3 && sw == 1 && no_dilation &&
             pads_equal) {
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

REGISTER_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
