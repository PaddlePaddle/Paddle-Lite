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

#include "lite/kernels/arm/conv_transpose_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void Conv2DTransposeCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int win = x_dims[3];  // nchw
  int hin = x_dims[2];
  int chin = x_dims[1];
  int num = x_dims[0];
  int wout = o_dims[3];
  int hout = o_dims[2];
  int chout = o_dims[1];
  int kw = w_dims[3];  // oihw
  int kh = w_dims[2];
  int group = param.groups;

  // deconv weights layout: chin * chout * kh * kw
  int m = chout * kw * kh / group;
  int n = hin * win;
  int k = chin / group;

  workspace_size_ = group * m * n * sizeof(float);

  auto& ctx = this->ctx_->template As<ARMContext>();
  auto dilations = *param.dilations;
  bool ks_equal = (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  depthwise_ =
      (param.groups == chin && chin == chout && ks_equal && no_dilation);
  bool depth_wise_s1 =
      depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);
  bool depth_wise_s2 =
      depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);
  if (!depth_wise_s1 && !depth_wise_s2) {
    lite::Tensor tmp_weights;
    lite::arm::math::prepackA(
        &tmp_weights, *(param.filter), 1.f, m, k, group, true, &ctx);
    param.filter->Resize(tmp_weights.dims());
    param.filter->CopyDataFrom(tmp_weights);
    param.filter->Resize(w_dims);
  }
  is_first_epoch_ = false;
}

void Conv2DTransposeCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  auto w_dims = param.filter->dims();
  int num = x_dims[0];
  int chin = x_dims[1];
  int hin = x_dims[2];
  int win = x_dims[3];
  int chout = o_dims[1];
  int hout = o_dims[2];
  int wout = o_dims[3];
  int kw = w_dims[3];  // oihw
  int kh = w_dims[2];
  int group = param.groups;
  bool flag_bias = (param.bias != nullptr);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  int m = chout * kw * kh / group;
  int n = hin * win;
  int k = chin / group;

  bool pads_equal =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);

  int group_size_in = win * hin * chin / group;
  int group_size_out = wout * hout * chout / group;
  int group_size_coldata = m * n;

  bool pads_all_qual = pads_equal && (paddings[0] == paddings[2]);
  int hblock = lite::arm::math::get_hblock(&ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_weights = ((m_roundup * k + 15) / 16) * 16;
  bool flag_1x1s1p1 = (kw == 1) && (kh == 1) && (param.strides[0] == 1) &&
                      (param.strides[1] == 1) && pads_all_qual &&
                      (paddings[0] == 0) && (dilations[0] == 1) &&
                      (dilations[1] == 1);
  ctx.ExtendWorkspace(sizeof(float) * group * m * n);

  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  bool depthwise_s1 =
      depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);
  bool depthwise_s2 =
      depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);
  for (int i = 0; i < num; i++) {
    const float* din_batch = din + i * chin * hin * win;
    float* dout_batch = dout + i * chout * hout * wout;
    if (depthwise_s1) {
      lite::arm::math::conv_transpose_depthwise_s1<float>(din_batch,
                                                          weights,
                                                          chout,
                                                          hout,
                                                          wout,
                                                          kh,
                                                          kw,
                                                          paddings[0],
                                                          paddings[1],
                                                          paddings[2],
                                                          paddings[3],
                                                          dilations[0],
                                                          dilations[1],
                                                          dout_batch,
                                                          &ctx);
    } else if (depthwise_s2) {
      lite::arm::math::conv_transpose_depthwise_s2<float>(din_batch,
                                                          weights,
                                                          chout,
                                                          hout,
                                                          wout,
                                                          kh,
                                                          kw,
                                                          paddings[0],
                                                          paddings[1],
                                                          paddings[2],
                                                          paddings[3],
                                                          dilations[0],
                                                          dilations[1],
                                                          dout_batch,
                                                          &ctx);
    } else {
      float* col_data = static_cast<float*>(ctx.workspace_data<float>()) +
                        ctx.llc_size() / sizeof(float);
      if (flag_1x1s1p1) {
        col_data = dout_batch;
      }
      for (int g = 0; g < group; g++) {
        const float* din_group = din_batch + g * group_size_in;
        const float* weights_group = weights + g * group_size_weights;
        float* coldata_group = col_data + g * group_size_coldata;
        if (flag_bias) {
          act_param.has_active = false;
        }
        lite::arm::math::sgemm_prepack(false,
                                       m,
                                       n,
                                       k,
                                       weights_group,
                                       din_group,
                                       n,
                                       0.f,
                                       coldata_group,
                                       n,
                                       nullptr,
                                       false,
                                       act_param,
                                       &ctx);
      }
      if (!flag_1x1s1p1) {
        lite::arm::math::col2im<float>(col_data,
                                       chout,
                                       hout,
                                       wout,
                                       kh,
                                       kw,
                                       paddings[0],
                                       paddings[1],
                                       paddings[2],
                                       paddings[3],
                                       param.strides[0],
                                       param.strides[1],
                                       dilations[0],
                                       dilations[1],
                                       dout_batch);
      }
    }
    if (flag_bias) {
      act_param.has_active = has_act;
      lite::arm::math::fill_bias_act<float>(
          dout_batch,
          static_cast<const float*>(param.bias->data<float>()),
          chout,
          wout * hout,
          flag_bias,
          &act_param);
    }
  }
}
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d_transpose,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::Conv2DTransposeCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
