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

#include "lite/kernels/x86/conv_transpose_compute.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/conv2d_transpose.h"
#include "lite/backends/x86/math/fill_bias_activate.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {
#define INIT_PARAM                                   \
  auto& param = this->Param<param_t>();              \
  auto x_dims = param.x->dims();                     \
  auto w_dims = param.filter->dims();                \
  auto o_dims = param.output->dims();                \
  int win = x_dims[3];                               \
  int hin = x_dims[2];                               \
  int chin = x_dims[1];                              \
  int num = x_dims[0];                               \
  int wout = o_dims[3];                              \
  int hout = o_dims[2];                              \
  int chout = o_dims[1];                             \
  int kw = w_dims[3];                                \
  int kh = w_dims[2];                                \
  int group = param.groups;                          \
  /* deconv weights layout: chin * chout * kh * kw*/ \
  int m = chout * kw * kh / group;                   \
  int n = hin * win;                                 \
  int k = chin / group;

#define DEPTHWISE_PARAM                                                   \
  auto dilations = *param.dilations;                                      \
  bool ks_equal = (param.strides[0] == param.strides[1]) && (kw == kh);   \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);          \
  depthwise_ =                                                            \
      (param.groups == chin && chin == chout && ks_equal && no_dilation); \
  bool depth_wise_s1 =                                                    \
      depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);     \
  bool depth_wise_s2 =                                                    \
      depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);

#define DEPTHWISE_FUNCS                                                    \
  din_batch, weights, chout, hout, wout, kh, kw, paddings[0], paddings[1], \
      paddings[2], paddings[3], dilations[0], dilations[1], dout_batch, &ctx

template <>
void Conv2DTransposeCompute<PRECISION(kFloat),
                            PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int win = x_dims[3];
  int hin = x_dims[2];
  int chin = x_dims[1];
  int chout = o_dims[1];
  int kw = w_dims[3];
  int kh = w_dims[2];
  int m = chout * kw * kh / param.groups;
  int n = hin * win;

  workspace_size_ = param.groups * m * n * sizeof(float);
  auto dilations = *param.dilations;
  bool ks_equal = (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  depthwise_ =
      (param.groups == chin && chin == chout && ks_equal && no_dilation);
  is_first_epoch_ = false;
}

PROFILE_INFO(kFloat, kFloat)
template <>
void Conv2DTransposeCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& ctx = this->ctx_->template As<X86Context>();
  INIT_PARAM
  bool flag_bias = (param.bias != nullptr);
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  bool pads_equal =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);

  int group_size_in = win * hin * chin / group;
  int group_size_weights = chin / group * chout / group * kw * kh;
  int group_size_coldata = m * n;
  bool pads_all_qual = pads_equal && (paddings[0] == paddings[2]);
  bool flag_1x1s1p1 = (kw == 1) && (kh == 1) && (param.strides[0] == 1) &&
                      (param.strides[1] == 1) && pads_all_qual &&
                      (paddings[0] == 0) && (dilations[0] == 1) &&
                      (dilations[1] == 1);
  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  auto act_param = param.activation_param;
  bool depthwise_s1 =
      depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);
  bool depthwise_s2 =
      depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  float* col_data = nullptr;

  if (!flag_1x1s1p1) {
    int col_size = param.groups * group_size_coldata;
    col_data = static_cast<float*>(
        TargetMalloc(TARGET(kX86), col_size * sizeof(float)));
  }

  for (int i = 0; i < num; i++) {
    const float* din_batch = din + i * chin * hin * win;
    float* dout_batch = dout + i * chout * hout * wout;

    if (depthwise_s1) {
      lite::x86::math::conv_transpose_depthwise_s1(DEPTHWISE_FUNCS);
    } else if (depthwise_s2) {
      lite::x86::math::conv_transpose_depthwise_s2(DEPTHWISE_FUNCS);
    } else {
      paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(ctx);
      if (flag_1x1s1p1) {
        col_data = dout_batch;
      }
      for (int g = 0; g < group; g++) {
        const float* din_group = din_batch + g * group_size_in;
        const float* weights_group = weights + g * group_size_weights;
        float* coldata_group = col_data + g * group_size_coldata;
        matmul.GEMM<float>(true,
                           false,
                           m,
                           n,
                           k,
                           1.f,
                           weights_group,
                           m,
                           din_group,
                           n,
                           0.f,
                           coldata_group,
                           n);
      }
      if (!flag_1x1s1p1) {
        lite::x86::math::col2im(col_data,
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
    // bias and activate
    lite::x86::math::fill_bias_act(
        dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
  }
  if (!flag_1x1s1p1) TargetFree(TARGET(kX86), col_data);
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::x86::Conv2DTransposeCompute<PRECISION(kFloat),
                                                           PRECISION(kFloat)>
    ConvTransFp32;

REGISTER_LITE_KERNEL(conv2d_transpose, kX86, kFloat, kNCHW, ConvTransFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();
