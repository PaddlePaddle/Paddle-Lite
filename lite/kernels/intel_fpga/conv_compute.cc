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

#include "lite/kernels/intel_fpga/conv_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/intel_fpga/conv_depthwise.h"
#include "lite/kernels/intel_fpga/conv_gemmlike.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {
#define PARAM_INIT                                                           \
  auto& param = this->Param<param_t>();                                      \
  auto w_dims = param.filter->dims();                                        \
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
    VLOG(3) << "[IntelFPGA] invoking depthwise conv";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "[IntelFPGA] invoking common conv";
  }
  if (!arm_cxt_) {
    arm_cxt_ = ContextScheduler::Global().NewContext(TargetType::kARM);
  }
  impl_->SetContext(std::move(arm_cxt_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::intel_fpga::ConvCompute<PRECISION(kFloat),
                                                       PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(conv2d, kIntelFPGA, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kIntelFPGA, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
