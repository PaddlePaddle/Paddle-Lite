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

#include "lite/kernels/arm/deformable_conv_compute.h"
#include <cmath>
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
void DeformableConvCompute<PRECISION(kFloat),
                           PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

static inline float deformable_bilinear(const float* bottom_data,
                                        const int height,
                                        const int width,
                                        float h,
                                        float w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = static_cast<float>(h_low);
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = static_cast<float>(w_low);
  } else {
    w_high = w_low + 1;
  }
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh;
  float hw = 1 - lw;
  float v1 = bottom_data[h_low * width + w_low];
  float v2 = bottom_data[h_low * width + w_high];
  float v3 = bottom_data[h_high * width + w_low];
  float v4 = bottom_data[h_high * width + w_high];
  float w1 = hh * hw;
  float w2 = hh * lw;
  float w3 = lh * hw;
  float w4 = lh * lw;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
void DeformableConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  // basic implement
  // param.x shape [n, cin, hin, win];
  // param.offset shape [n, 2 * deformabel_group * kw * kh, hin, win]
  // param.mask shape [n, deformabel_group * kw * kh, hin, win]
  // param.filter shape [cout, cin/group, kw, kh]
  // param.output shape [n, cout, hout, wout]
  // deformable_group == group
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* in_data = param.x->data<float>();
  const auto* filter_data = param.conv_param.filter->data<float>();
  const auto* offset_data = param.offset->data<float>();
  const auto* mask_data = param.mask->data<float>();
  float* out_data = param.output->mutable_data<float>();

  auto in_dims = param.x->dims();
  auto filter_dims = param.conv_param.filter->dims();
  auto out_dims = param.output->dims();
  auto stride = param.conv_param.strides;
  auto paddings = *param.conv_param.paddings;
  auto dilation = *param.conv_param.dilations;
  auto group = param.conv_param.groups;
  auto deformable_group = param.deformable_groups;

  auto num = in_dims[0];
  auto cin = in_dims[1];
  auto hin = in_dims[2];
  auto win = in_dims[3];
  auto cout = filter_dims[0];
  auto kh = filter_dims[2];
  auto kw = filter_dims[3];
  auto hout = out_dims[2];
  auto wout = out_dims[3];
  bool is_bias = param.conv_param.bias ? true : false;
  const float* bias =
      param.conv_param.bias ? param.conv_param.bias->data<float>() : nullptr;

  auto in_c_group = cin / group;
  auto out_c_group = cout / group;
  float alpha = 1.f;
  const float beta = 0.f;
  int in_size = hin * win;
  int out_size = hout * wout;
  int c_in_size = cin * in_size;
  int c_out_size = cout * out_size;
  int kernel_size = kw * kh;

  int col_size = num * cin * kernel_size * in_size;
  auto offset_in_size = 2 * group * kernel_size * in_size;
  float* col_data = new float[col_size];
  for (int n = 0; n < num; n++) {
    for (int g = 0; g < group; ++g) {
      const float* offset_data_ptr =
          offset_data + n * offset_in_size + g * 2 * kernel_size * in_size;
      const float* in_data_offset =
          in_data + n * c_in_size + g * in_c_group * in_size;
      float* col_data_g = col_data + n * c_in_size * kernel_size +
                          g * in_c_group * kernel_size * in_size;
      for (int ic = 0; ic < in_c_group; ++ic) {
        const float* in_data_ch = in_data_offset + ic * in_size;
        float* col_data_ch = col_data_g + ic * kernel_size * in_size;
        for (int fh = 0; fh < kh; fh++) {
          for (int fw = 0; fw < kw; fw++) {
            const float* offset_data_ptr_h =
                offset_data_ptr + (2 * (fh * kw + fw)) * out_size;
            const float* offset_data_ptr_w =
                offset_data_ptr + (2 * (fh * kw + fw) + 1) * out_size;
            float* col_data_g_ksize = col_data_ch + (fh * kw + fw) * in_size;
            for (int ih = 0; ih < hin; ih++) {
              const float* offset_data_ptr_h_w = offset_data_ptr_h + ih * wout;
              const float* offset_data_ptr_w_w = offset_data_ptr_w + ih * wout;
              float* col_data_g_ksize_h = col_data_g_ksize + ih * win;
              for (int iw = 0; iw < win; iw++) {
                const float offset_h = *offset_data_ptr_h_w++;
                const float offset_w = *offset_data_ptr_w_w++;
                const float im_w =
                    iw * stride[1] - paddings[2] + kw * dilation[1] + offset_w;
                const float im_h =
                    ih * stride[0] - paddings[0] + kh * dilation[0] + offset_h;
                if (im_h >= 0 && im_h < hin && im_w >= 0 && im_w < win) {
                  float val =
                      deformable_bilinear(in_data_ch, hin, win, im_h, im_w);

                  if (param.modulated) {
                    // use mask
                    const float* mask_ptr =
                        mask_data + n * group * kernel_size * in_size +
                        g * kernel_size * in_size +
                        (fh * kw + fw) * hout * wout + ih * win + iw;
                    val *= mask_ptr[0];
                  }
                  *col_data_g_ksize_h++ = val;
                } else {
                  *col_data_g_ksize_h++ = 0.0;
                }
              }
            }
          }
        }
      }
    }
  }
  // convolution
  int m = cout / group;
  int n = hout * wout;
  int k = cin * kernel_size / group;
  int weights_size_per_group = m * k;
  if (flag_trans_weights_) {
    filter_data = weights_.data<float>();
  }
  for (int b = 0; b < num; ++b) {
    for (int g = 0; g < group; ++g) {
      float* dout_group = out_data + (b * cout + g * m) * out_size;
      const float* din_group =
          col_data + (b * cin + g * in_c_group) * in_size * kernel_size;
      const float* weights_group = filter_data + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      if (n == 1) {
        lite::arm::math::sgemv(
            weights_group,
            din_group,
            dout_group,
            false,
            m,
            k,
            0.f,
            is_bias,
            bias_group,
            param.conv_param.activation_param.has_active,
            param.conv_param.activation_param.active_type,
            &ctx,
            param.conv_param.activation_param.Relu_clipped_coef,
            param.conv_param.activation_param.Leaky_relu_alpha);
      } else {
        int ldb = n;
        lite::arm::math::sgemm_prepack(false,
                                       m,
                                       n,
                                       k,
                                       weights_group,
                                       din_group,
                                       ldb,
                                       0.f,
                                       dout_group,
                                       n,
                                       bias_group,
                                       is_bias,
                                       param.conv_param.activation_param,
                                       &ctx);
      }
    }
  }
  delete[] col_data;
}
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::DeformableConvCompute<PRECISION(kFloat),
                                                          PRECISION(kFloat)>
    DeformableConvFp32;

REGISTER_LITE_KERNEL(
    deformable_conv, kARM, kFloat, kNCHW, DeformableConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
