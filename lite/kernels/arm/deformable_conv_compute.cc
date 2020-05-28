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
void DeformableConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

static inline float deformable_bilinear(const float* bottom_data, const int data_width,
                          const int height, const int width, float h, float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (float) h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (float) w_low;
    } else {
        w_high = w_low + 1;
    }
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh;
    float hw = 1 - lw;
    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
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
  const auto* filter_data = param.filter->data<float>();
  const auto* offset_data = param.offset->data<float>();
  const auto* mask_data = param.mask->data<float>();
  float* out_data = param.output->mutable_data<float>();

  auto in_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto out_dims = param.output->dims();
  auto stride = param.strides;
  auto paddings = param.paddings;
  auto dilation = param.dilations;
  auto group = param.groups;
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
  bool is_bias = param.bias ? true : false;
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;

  auto in_c_group = cin / group;
  auto out_c_group = cout / group;
  float alpha = 1.f;
  const float beta = 0.f;
  int in_size = hin * win;
  int out_size = hout * wout;
  int c_in_size = cin * in_size;
  int c_out_size = cout * out_size;
  int kernel_size = kw * kh;

  //lite::Tensor col;
  //col.Resize({num, cin * kernel_size, hin, win});
  int col_size = num * cin * kernel_size * in_size;
//   float* tmp_work_space =
//       ctx.workspace_data<float>() + ctx.llc_size() / sizeof(float);
  auto offset_in_size = 2 * group * kernel_size * in_size; 
  //auto* col_data = col.mutable_data<float>();
  float* col_data = new float[col_size];
  memset(col_data, 0.0, sizeof(float) * col_size);
  for (int n = 0; n < num; n++) {
      for (int g = 0; g < group; ++g) {
          for (int ic = 0; ic < in_c_group; ++ic) {
              for (int fh = 0; fh < kh; fh++) {
                  for (int fw = 0; fw < kw; fw++) {
                      for (int ih = 0; ih < hin; ih++) {
                          for (int iw = 0; iw < win; iw++) {
                            const float* offset_data_ptr = offset_data
                                                           + n * offset_in_size
                                                           + g * 2 * kernel_size * in_size;
                            const int data_offset_h_ptr = ((2 * (fh * kw + fw))
                                                            * hout + ih) * wout + iw;
                            const int data_offset_w_ptr = ((2 * (fh * kw + fw) + 1)
                                                            * hout + ih) * wout + iw;
                            const float offset_h = offset_data_ptr[data_offset_h_ptr];
                            const float offset_w = offset_data_ptr[data_offset_w_ptr];
                            const float im_w = iw * stride[1] - paddings[1] + kw * dilation[1] + offset_w;
                            const float im_h = ih * stride[0] - paddings[0] + kh * dilation[0] + offset_h;
                            if (im_h > -1 && im_h < hin && im_w > -1 && im_w < win) {
                                // get data
                                const float map_h = kh * dilation[0] + offset_h;
                                const float map_w = kw * dilation[1] + offset_w;
                                const int cur_height = hin - (ih * stride[0] - paddings[0]);
                                const int cur_width = win - (iw * stride[1] - paddings[1]);

                                const float* in_data_offset = in_data + n * c_in_size
                                                                + (g * in_c_group + ic) * hin * win 
                                                                + (ih * stride[0] - paddings[0]) * win 
                                                                + (iw * stride[1] - paddings[1]);

                                int out_idx = n * c_in_size * kernel_size + g * in_c_group * kernel_size * in_size
                                                + ic * kernel_size * in_size + ((fh * kw + fw) * ih + ih) * win + iw;
                                float val = deformable_bilinear(in_data_offset, 
                                                                win, cur_height, cur_width, map_h, map_w);
                                //printf("map_h: %f, map_w: %f, cur_height: %d, cur_width: %d \n", map_h, map_w, cur_height, cur_width);
                                // printf("val: %f \n", val);
                                if (param.modulated) {
                                   // use mask
                                   const float* mask_ptr = mask_data + n * group * kernel_size * in_size
                                                           + g * kernel_size * in_size
                                                           + (fh * kw + fw) * hout * wout
                                                           + ih * win + iw; 
                                   val *= mask_ptr[0];
                                }
                                //printf("val: %f \n", val);
                                col_data[out_idx] = val;
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
  int k = cin  *  kernel_size / group;
  int weights_size_per_group = m * k;
  if (flag_trans_weights_) {
    filter_data = weights_.data<float>();
  }
  for (int b = 0; b < num; ++b) {
      for (int g = 0; g < group; ++g) {
          float* dout_group = out_data + (b * cout + g * m) * out_size;
          const float* din_group = col_data + (b * cin + g * in_c_group) * in_size;
          const float* weights_group = filter_data + g * weights_size_per_group;
          const float* bias_group = bias + g * m;
          if (n == 1) {
              lite::arm::math::sgemv(weights_group,
                    din_group,
                    dout_group,
                    false,
                    m,
                    k,
                    is_bias,
                    bias_group,
                    param.activation_param.has_active,
                    param.activation_param.active_type,
                    &ctx,
                    param.activation_param.Relu_clipped_coef,
                    param.activation_param.Leaky_relu_alpha);
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
                            param.activation_param,
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

REGISTER_LITE_KERNEL(deformconv2d, kARM, kFloat, kNCHW, DeformableConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
