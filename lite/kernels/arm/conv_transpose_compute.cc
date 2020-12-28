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
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <>
void Conv2DTransposeCompute<PRECISION(kFloat),
                            PRECISION(kFloat)>::PrepareForRun() {
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
template <>
void Conv2DTransposeCompute<PRECISION(kInt8),
                            PRECISION(kFloat)>::PrepareForRun() {
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

  workspace_size_ = 2 * group * m * n * sizeof(int32_t);

  auto& ctx = this->ctx_->template As<ARMContext>();
  lite::Tensor tmp_weights;
  lite::arm::math::prepackA_int8(
      &tmp_weights, *(param.filter), m, k, group, true, &ctx);
  param.filter->Resize(tmp_weights.dims());
  param.filter->CopyDataFrom(tmp_weights);
  param.filter->Resize(w_dims);
  // update scale
  w_scale_ = param.weight_scale;
  auto cout = w_dims[1] * group;
  if (w_scale_.size() != 1 && w_scale_.size() != cout) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < cout - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  for (auto& ws : w_scale_) {
    ws *= input_scale;
  }
}
template <>
void Conv2DTransposeCompute<PRECISION(kInt8),
                            PRECISION(kInt8)>::PrepareForRun() {
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

  // col_out(m*n*group) + gemm_out(m*n*group)
  workspace_size_ = 2 * group * m * n * sizeof(int32_t);

  auto& ctx = this->ctx_->template As<ARMContext>();
  lite::Tensor tmp_weights;
  lite::arm::math::prepackA_int8(
      &tmp_weights, *(param.filter), m, k, group, true, &ctx);
  param.filter->Resize(tmp_weights.dims());
  param.filter->CopyDataFrom(tmp_weights);
  param.filter->Resize(w_dims);
  // update scale
  w_scale_ = param.weight_scale;
  auto cout = w_dims[1] * group;
  if (w_scale_.size() != 1 && w_scale_.size() != cout) {
    LOG(FATAL) << "weights scale size must equal to filter size, scales size: "
               << w_scale_.size() << ", cout: " << cout;
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < cout - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  for (auto& ws : w_scale_) {
    ws = ws * input_scale / output_scale;
  }
  //!  update bias
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = param.bias->data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / param.output_scale;
    }
    flag_trans_bias_ = true;
  }
  //! update relu6 parameter
  param.activation_param.Relu_clipped_coef =
      param.activation_param.Relu_clipped_coef / param.output_scale;
  param.activation_param.Leaky_relu_alpha =
      param.activation_param.Leaky_relu_alpha / param.output_scale;
}

template <>
void Conv2DTransposeCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
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

  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  bool depthwise_s1 =
      depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);
  bool depthwise_s2 =
      depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);
  bool bias_act = flag_bias || has_act;
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
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
      if (bias_act) {
        lite::arm::math::fill_bias_act<float>(
            dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
      }
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
      if (bias_act) {
        lite::arm::math::fill_bias_act<float>(
            dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
      }
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
}
template <>
void Conv2DTransposeCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
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
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }

  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<int8_t>();
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  int32_t* workspace_ptr =
      static_cast<int32_t*>(ctx.workspace_data<int32_t>()) +
      ctx.llc_size() / sizeof(int32_t);
  int offset = group * m * n;
  for (int i = 0; i < num; i++) {
    const int8_t* din_batch = din + i * chin * hin * win;
    float* dout_batch = dout + i * chout * hout * wout;
    int32_t* dout_batch_int32 = workspace_ptr + offset;
    int32_t* col_data = static_cast<int32_t*>(ctx.workspace_data<int32_t>()) +
                        ctx.llc_size() / sizeof(int32_t);
    if (flag_1x1s1p1) {
      col_data = dout_batch_int32;
    }
    for (int g = 0; g < group; g++) {
      const int8_t* din_group = din_batch + g * group_size_in;
      const int8_t* weights_group = weights + g * group_size_weights;
      const float* scale_group = w_scale_.data() + g * m;
      int32_t* coldata_group = col_data + g * group_size_coldata;
      if (flag_bias) {
        act_param.has_active = false;
      }
      lite::arm::math::gemm_prepack_int8<int32_t>(weights_group,
                                                  din_group,
                                                  nullptr,
                                                  coldata_group,
                                                  m,
                                                  n,
                                                  k,
                                                  false,
                                                  false,
                                                  scale_group,
                                                  act_param,
                                                  &ctx);
    }
    if (!flag_1x1s1p1) {
      lite::arm::math::col2im<int>(col_data,
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
                                   dout_batch_int32);
    }
    act_param.has_active = has_act;
    // int32 -> fp32 int32*scale + bias
    lite::arm::math::fill_bias_act_calib<float>(dout_batch,
                                                dout_batch_int32,
                                                bias,
                                                w_scale_.data(),
                                                chout,
                                                wout * hout,
                                                flag_bias,
                                                &act_param);
  }
}
template <>
void Conv2DTransposeCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
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
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }

  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<int8_t>();
  auto weights = param.filter->data<int8_t>();
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  int32_t* workspace_ptr =
      static_cast<int32_t*>(ctx.workspace_data<int32_t>()) +
      ctx.llc_size() / sizeof(int32_t);
  int offset = group * m * n;
  for (int i = 0; i < num; i++) {
    const int8_t* din_batch = din + i * chin * hin * win;
    int8_t* dout_batch = dout + i * chout * hout * wout;
    int32_t* col_data = workspace_ptr;
    int32_t* dout_batch_int32 = workspace_ptr + offset;
    if (flag_1x1s1p1) {
      col_data = dout_batch_int32;
    }
    for (int g = 0; g < group; g++) {
      const int8_t* din_group = din_batch + g * group_size_in;
      const int8_t* weights_group = weights + g * group_size_weights;
      const float* scale_group = w_scale_.data() + g * m;
      int32_t* coldata_group = col_data + g * group_size_coldata;
      if (flag_bias) {
        act_param.has_active = false;
      }
      lite::arm::math::gemm_prepack_int8<int32_t>(weights_group,
                                                  din_group,
                                                  nullptr,
                                                  coldata_group,
                                                  m,
                                                  n,
                                                  k,
                                                  false,
                                                  false,
                                                  scale_group,
                                                  act_param,
                                                  &ctx);
    }
    if (!flag_1x1s1p1) {
      lite::arm::math::col2im<int>(col_data,
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
                                   dout_batch_int32);
    }
    // int32 -> int8 int32*scale + bias
    act_param.has_active = has_act;
    lite::arm::math::fill_bias_act_calib<int8_t>(dout_batch,
                                                 dout_batch_int32,
                                                 bias,
                                                 w_scale_.data(),
                                                 chout,
                                                 wout * hout,
                                                 flag_bias,
                                                 &act_param);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::Conv2DTransposeCompute<PRECISION(kFloat),
                                                           PRECISION(kFloat)>
    ConvTransFp32;
typedef paddle::lite::kernels::arm::Conv2DTransposeCompute<PRECISION(kInt8),
                                                           PRECISION(kFloat)>
    ConvTranInt8_Fp32;
typedef paddle::lite::kernels::arm::Conv2DTransposeCompute<PRECISION(kInt8),
                                                           PRECISION(kInt8)>
    ConvTranInt8_Int8;
REGISTER_LITE_KERNEL(conv2d_transpose, kARM, kFloat, kNCHW, ConvTransFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d_transpose, kARM, kInt8, kNCHW, ConvTranInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d_transpose, kARM, kInt8, kNCHW, ConvTranInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();
