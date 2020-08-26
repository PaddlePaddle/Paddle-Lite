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

#include <arm_neon.h>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/conv_depthwise.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
void conv_depthwise_5x5s2_bias(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int pad_top,
                               int pad_bottom,
                               int pad_left,
                               int pad_right,
                               ARMContext* ctx);
void conv_depthwise_5x5s2_bias_relu(float* dout,
                                    const float* din,
                                    const float* weights,
                                    const float* bias,
                                    bool flag_bias,
                                    int num,
                                    int chin,
                                    int hin,
                                    int win,
                                    int hout,
                                    int wout,
                                    int pad_top,
                                    int pad_bottom,
                                    int pad_left,
                                    int pad_right,
                                    ARMContext* ctx);
void conv_depthwise_5x5s2_bias_relu6(float* dout,
                                     const float* din,
                                     const float* weights,
                                     const float* bias,
                                     const float* six,
                                     bool flag_bias,
                                     int num,
                                     int chin,
                                     int hin,
                                     int win,
                                     int hout,
                                     int wout,
                                     int pad_top,
                                     int pad_bottom,
                                     int pad_left,
                                     int pad_right,
                                     ARMContext* ctx);
void conv_depthwise_5x5s2_bias_leakyRelu(float* dout,
                                         const float* din,
                                         const float* weights,
                                         const float* bias,
                                         const float* scale,
                                         bool flag_bias,
                                         int num,
                                         int chin,
                                         int hin,
                                         int win,
                                         int hout,
                                         int wout,
                                         int pad_top,
                                         int pad_bottom,
                                         int pad_left,
                                         int pad_right,
                                         ARMContext* ctx);
void conv_depthwise_5x5s2_fp32(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int pad_top,
                               int pad_bottom,
                               int pad_left,
                               int pad_right,
                               const operators::ActivationParam& act_param,
                               ARMContext* ctx) {
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;
  float tmp = act_param.Relu_clipped_coef;
  float ss = act_param.Leaky_relu_alpha;
  float vsix[4] = {tmp, tmp, tmp, tmp};
  float vscale[4] = {ss, ss, ss, ss};
  if (has_active) {
    switch (act_type) {
      case lite_api::ActivationType::kRelu:
        conv_depthwise_5x5s2_bias_relu(dout,
                                       din,
                                       weights,
                                       bias,
                                       flag_bias,
                                       num,
                                       chin,
                                       hin,
                                       win,
                                       hout,
                                       wout,
                                       pad_top,
                                       pad_bottom,
                                       pad_left,
                                       pad_right,
                                       ctx);
        break;
      case lite_api::ActivationType::kRelu6:
        conv_depthwise_5x5s2_bias_relu6(dout,
                                        din,
                                        weights,
                                        bias,
                                        vsix,
                                        flag_bias,
                                        num,
                                        chin,
                                        hin,
                                        win,
                                        hout,
                                        wout,
                                        pad_top,
                                        pad_bottom,
                                        pad_left,
                                        pad_right,
                                        ctx);
        break;
      case lite_api::ActivationType::kLeakyRelu:
        conv_depthwise_5x5s2_bias_leakyRelu(dout,
                                            din,
                                            weights,
                                            bias,
                                            vscale,
                                            flag_bias,
                                            num,
                                            chin,
                                            hin,
                                            win,
                                            hout,
                                            wout,
                                            pad_top,
                                            pad_bottom,
                                            pad_left,
                                            pad_right,
                                            ctx);
        break;
      default:
        LOG(FATAL) << "this act_type: " << ActivationTypeToStr(act_type)
                   << " fuse not support";
    }
  } else {
    conv_depthwise_5x5s2_bias(dout,
                              din,
                              weights,
                              bias,
                              flag_bias,
                              num,
                              chin,
                              hin,
                              win,
                              hout,
                              wout,
                              pad_top,
                              pad_bottom,
                              pad_left,
                              pad_right,
                              ctx);
  }
}
// clang-format off
#ifdef __aarch64__
#define COMPUTE_ONE_LINE_S2_PRE                        \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1  {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},   [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_TWO_LINE_S2_PRE                        \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},   [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_THREE_LINE_S2_PRE                      \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1  {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_FOUR_LINE_S2_PRE                       \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1  {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr3]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr3]]\n"     /*891011*/      \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr3].s[0]\n"  /*0246*wr3[0]*/ \
  "fmla v15.4s, v10.4s, %[wr3].s[1]\n" /*1357*wr3[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1  {v11.4s}, [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v16.4s, v12.4s, %[wr3].s[2]\n" /*2468*wr3[2]*/ \
  "fmla v15.4s, v13.4s, %[wr3].s[3]\n" /*3579*wr3[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_FIVE_LINE_S2                           \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr3]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr3]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr3].s[0]\n"  /*0246*wr3[0]*/ \
  "fmla v15.4s, v10.4s, %[wr3].s[1]\n" /*1357*wr3[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr4]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr3].s[2]\n" /*2468*wr3[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr4]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr3].s[3]\n" /*3579*wr3[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr4].s[0]\n"  /*0246*wr4[0]*/ \
  "fmla v16.4s, v10.4s, %[wr4].s[1]\n" /*1357*wr4[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr4].s[2]\n" /*2468*wr4[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr4].s[3]\n" /*3579*wr4[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_FIVE_LINE_S2_OUT2                      \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v17.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v17.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "fmul v18.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr3]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "fmla v17.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr3]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "fmla v18.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "fmla v17.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr3].s[0]\n"  /*0246*wr3[0]*/ \
  "fmla v18.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr3].s[1]\n" /*1357*wr3[1]*/ \
  "fmla v17.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr4]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr3].s[2]\n" /*2468*wr3[2]*/ \
  "fmla v18.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr4]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr3].s[3]\n" /*3579*wr3[3]*/ \
  "fmla v17.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "fmla v18.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr4].s[0]\n"  /*0246*wr4[0]*/ \
  "fmla v17.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr4].s[1]\n" /*1357*wr4[1]*/ \
  "fmla v18.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr5]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr4].s[2]\n" /*2468*wr4[2]*/ \
  "fmla v17.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr5]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr4].s[3]\n" /*3579*wr4[3]*/ \
  "fmla v18.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "fmla v17.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v18.4s, v9.4s, %[wr3].s[0]\n"  /*0246*wr3[0]*/ \
  "fmla v17.4s, v10.4s, %[wr3].s[1]\n"  /*1357*wr3[1]*/\
  "ld2 {v9.4s, v10.4s}, [%[din_ptr6]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1 {v11.4s},  [%[din_ptr6]]\n"     /*891011*/      \
  "fmla v18.4s, v12.4s, %[wr3].s[2]\n" /*2468*wr3[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr6]]\n"     /*891011*/      \
  "fmla v17.4s, v13.4s, %[wr3].s[3]\n" /*3579*wr3[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v18.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v17.4s, v9.4s, %[wr4].s[0]\n"  /*0246*wr4[0]*/ \
  "fmla v18.4s, v10.4s, %[wr4].s[1]\n"  /*1357*wr4[1]*/\
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v17.4s, v12.4s, %[wr4].s[2]\n" /*2468*wr4[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v18.4s, v13.4s, %[wr4].s[3]\n" /*3579*wr4[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v17.4s, v14.4s, %[wr6].s[0]\n" /*46810*wr6[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"                      \
  "fadd v18.4s, v18.4s, v17.4s\n"

#define COMPUTE_ONE_LINE_S2_POST                       \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_TWO_LINE_S2_POST                       \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_THREE_LINE_S2_POST                     \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define COMPUTE_FOUR_LINE_S2_POST                      \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "1: \n"                                              \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmla v15.4s, v9.4s, %[wr0].s[0]\n"  /*0246*wr0[0]*/ \
  "fmul v16.4s, v10.4s, %[wr0].s[1]\n" /*1357*wr0[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr1]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr0].s[2]\n" /*2468*wr0[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr1]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr0].s[3]\n" /*3579*wr0[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[0]\n" /*46810*wr5[0]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr1].s[0]\n"  /*0246*wr1[0]*/ \
  "fmla v15.4s, v10.4s, %[wr1].s[1]\n" /*1357*wr1[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr2]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr1].s[2]\n" /*2468*wr1[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr2]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr1].s[3]\n" /*3579*wr1[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[1]\n" /*46810*wr5[1]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v15.4s, v9.4s, %[wr2].s[0]\n"  /*0246*wr2[0]*/ \
  "fmla v16.4s, v10.4s, %[wr2].s[1]\n" /*1357*wr2[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr3]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v15.4s, v12.4s, %[wr2].s[2]\n" /*2468*wr2[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr3]]\n"     /*891011*/      \
  "fmla v16.4s, v13.4s, %[wr2].s[3]\n" /*3579*wr2[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v15.4s, v14.4s, %[wr5].s[2]\n" /*46810*wr5[2]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fmla v16.4s, v9.4s, %[wr3].s[0]\n"  /*0246*wr3[0]*/ \
  "fmla v15.4s, v10.4s, %[wr3].s[1]\n" /*1357*wr3[1]*/ \
  "ld2 {v9.4s, v10.4s}, [%[din_ptr0]], #32\n"          \
  "mov v13.s[3], v11.s[1]\n"           /*3579*/        \
  "mov v14.s[3], v11.s[2]\n"           /*46810*/       \
  "fmla v16.4s, v12.4s, %[wr3].s[2]\n" /*2468*wr3[2]*/ \
  "ld1 {v11.4s},  [%[din_ptr0]]\n"     /*891011*/      \
  "fmla v15.4s, v13.4s, %[wr3].s[3]\n" /*3579*wr3[3]*/ \
  "ext v12.16b, v9.16b, v11.16b, #4\n" /*2468*/        \
  "fmla v16.4s, v14.4s, %[wr5].s[3]\n" /*46810*wr5[3]*/\
  "ext v13.16b, v10.16b, v11.16b, #4\n"/*3578*/        \
  "ext v14.16b, v9.16b, v11.16b, #8\n" /*4689*/        \
  "fadd v16.4s, v16.4s, v15.4s\n"
#define RESULT_S2                                      \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "st1 {v16.4s}, [%[dout_ptr]], #16\n"                 \
  "bne 1b"
#define RESULT_S2_RELU                                 \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fmax v16.4s, v16.4s, %[vzero].4s\n"                 \
  "st1 {v16.4s}, [%[dout_ptr]], #16\n"                 \
  "bne 1b"
#define RESULT_S2_RELU6                                \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fmax v16.4s, v16.4s, %[vzero].4s\n"                 \
  "fmin v16.4s, v16.4s, %[vsix].4s\n"                  \
  "st1 {v16.4s}, [%[dout_ptr]], #16\n"                 \
  "bne 1b"
#define RESULT_S2_LEAKY_RELU                           \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fcmge v17.4s, v16.4s,  %[vzero].4s\n"               \
  "fmul v18.4s, v16.4s, %[vscale].4s\n"                \
  "bif v16.16b, v18.16b, v17.16b\n"                    \
  "st1 {v16.4s}, [%[dout_ptr]], #16\n"                 \
  "bne 1b"
#define RESULT_S2_OUT2                                 \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "st1 {v16.4s}, [%[dout_ptr0]], #16\n"                \
  "ld1 {v17.4s}, [%[bias]]\n"                          \
  "st1 {v18.4s}, [%[dout_ptr1]], #16\n"                \
  "bne 1b"
#define RESULT_S2_RELU_OUT2                            \
  "fmax v16.4s, v16.4s, %[vzero].4s\n"                 \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fmax v18.4s, v18.4s, %[vzero].4s\n"                 \
  "ld1 {v17.4s}, [%[bias]]\n"                          \
  "st1 {v16.4s}, [%[dout_ptr0]], #16\n"                \
  "st1 {v18.4s}, [%[dout_ptr1]], #16\n"                \
  "bne 1b"
#define RESULT_S2_RELU6_OUT2                           \
  "fmax v16.4s, v16.4s, %[vzero].4s\n"                 \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fmax v18.4s, v18.4s, %[vzero].4s\n"                 \
  "ld1 {v17.4s}, [%[bias]]\n"                          \
  "fmin v16.4s, v16.4s, %[vsix].4s\n"                  \
  "fmin v18.4s, v18.4s, %[vsix].4s\n"                  \
  "st1 {v16.4s}, [%[dout_ptr0]], #16\n"                \
  "st1 {v18.4s}, [%[dout_ptr1]], #16\n"                \
  "bne 1b"
#define RESULT_S2_LEAKY_RELU_OUT2                      \
  "fcmge v19.4s, v16.4s,  %[vzero].4s\n"               \
  "fmul v20.4s, v16.4s, %[vscale].4s\n"                \
  "ld1 {v15.4s}, [%[bias]]\n"                          \
  "fcmge v21.4s, v18.4s,  %[vzero].4s\n"               \
  "fmul v22.4s, v18.4s, %[vscale].4s\n"                \
  "ld1 {v17.4s}, [%[bias]]\n"                          \
  "bif v16.16b, v20.16b, v19.16b\n"                    \
  "bif v18.16b, v22.16b, v21.16b\n"                    \
  "st1 {v16.4s}, [%[dout_ptr0]], #16\n"                \
  "st1 {v18.4s}, [%[dout_ptr1]], #16\n"                \
  "bne 1b"
#else
#define COMPUTE_ONE_LINE_S2_PRE                    \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_TWO_LINE_S2_PRE                    \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_THREE_LINE_S2_PRE                  \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_FOUR_LINE_S2_PRE                   \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr3]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr3]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr3][0]\n" /*0246*wr3[0]*/ \
  "vmla.f32 q15, q9, %e[wr3][1]\n" /*1357*wr3[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q14, q11, %f[wr3][0]\n" /*2468*wr3[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr3][1]\n" /*3579*wr3[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_FIVE_LINE_S2                       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr3]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr3]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr3][0]\n" /*0246*wr3[0]*/ \
  "vmla.f32 q15, q9, %e[wr3][1]\n" /*1357*wr3[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr4]]!\n"           \
  "vmla.f32 q14, q11, %f[wr3][0]\n" /*2468*wr3[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr4]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr3][1]\n" /*3579*wr3[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr4][0]\n" /*0246*wr4[0]*/ \
  "vmla.f32 q14, q9, %e[wr4][1]\n" /*1357*wr4[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q15, q11, %f[wr4][0]\n" /*2468*wr4[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr4][1]\n" /*3579*wr4[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_FIVE_LINE_S2_OUT2                  \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q15, q15, q14\n"                       \
  "vld1.f32 {d28-d29}, [%[bias]]\n"                \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vmla.f32 q15, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vmla.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vld2.f32 {d16-d19}, [%[din_ptr3]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vmla.f32 q14, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr3]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vmla.f32 q14, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vmla.f32 q15, q8, %e[wr3][0]\n" /*0246*wr3[0]*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q9, %e[wr3][1]\n" /*1357*wr3[1]*/ \
  "vmla.f32 q14, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr4]]!\n"           \
  "vmla.f32 q15, q11, %f[wr3][0]\n" /*2468*wr3[2]*/\
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr4]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vmla.f32 q14, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr3][1]\n" /*3579*wr3[3]*/\
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr4][0]\n" /*0246*wr4[0]*/ \
  "vmla.f32 q14, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vmla.f32 q15, q9, %e[wr4][1]\n" /*1357*wr4[1]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vld2.f32 {d16-d19}, [%[din_ptr5]]!\n"           \
  "vmla.f32 q15, q11, %f[wr4][0]\n" /*2468*wr4[2]*/\
  "vmla.f32 q14, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr5]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vmla.f32 q14, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr4][1]\n" /*3579*wr4[3]*/\
  "vmla.f32 q14, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr3][0]\n" /*0246*wr3[0]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vmla.f32 q14, q9, %e[wr3][1]\n" /*1357*wr3[1]*/ \
  "vld2.f32 {d16-d19}, [%[din_ptr6]]!\n"           \
  "vmla.f32 q14, q11, %f[wr3][0]\n" /*2468*wr3[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr6]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vmla.f32 q14, q12, %f[wr3][1]\n" /*3579*wr4[3]*/\
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr4][0]\n" /*0246*wr4[0]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vmla.f32 q14, q9, %e[wr4][1]\n" /*1357*wr4[1]*/ \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q14, q11, %f[wr4][0]\n" /*2468*wr4[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr6][0]\n"/*46810*wr6[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vmla.f32 q14, q12, %f[wr4][1]\n" /*3579*wr4[3]*/\
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/
#define COMPUTE_ONE_LINE_S2_POST                   \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_TWO_LINE_S2_POST                   \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_THREE_LINE_S2_POST                 \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][0]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define COMPUTE_FOUR_LINE_S2_POST                  \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "1: \n"                                          \
  "subs %[cnt], #1\n"                              \
  "vmla.f32 q15, q8, %e[wr0][0]\n" /*0246*wr0[0]*/ \
  "vmul.f32 q14, q9, %e[wr0][1]\n" /*1357*wr0[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr1]]!\n"           \
  "vmla.f32 q15, q11, %f[wr0][0]\n" /*2468*wr0[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr1]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %e[wr5][0]\n"/*46810*wr5[0]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr0][1]\n" /*3579*wr0[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr1][0]\n" /*0246*wr1[0]*/ \
  "vmla.f32 q15, q9, %e[wr1][1]\n" /*1357*wr1[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr2]]!\n"           \
  "vmla.f32 q14, q11, %f[wr1][0]\n" /*2468*wr1[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr2]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %e[wr5][1]\n"/*46810*wr5[1]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr1][1]\n" /*3579*wr1[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q15, q8, %e[wr2][0]\n" /*0246*wr2[0]*/ \
  "vmla.f32 q14, q9, %e[wr2][1]\n" /*1357*wr2[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr3]]!\n"           \
  "vmla.f32 q15, q11, %f[wr2][0]\n" /*2468*wr2[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr3]]\n"  /*810911*/\
  "vmla.f32 q14, q13, %f[wr5][0]\n"/*46810*wr5[2]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q15, q12, %f[wr2][1]\n" /*3579*wr2[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vmla.f32 q14, q8, %e[wr3][0]\n" /*0246*wr3[0]*/ \
  "vmla.f32 q15, q9, %e[wr3][1]\n" /*1357*wr3[1]*/ \
  "vext.f32 d24, d18, d19, #1\n"   /*13-35*/       \
  "vld2.f32 {d16-d19}, [%[din_ptr0]]!\n"           \
  "vmla.f32 q14, q11, %f[wr3][0]\n" /*2468*wr3[2]*/\
  "vld2.f32 {d20-d21}, [%[din_ptr0]]\n"  /*810911*/\
  "vmla.f32 q15, q13, %f[wr5][1]\n"/*46810*wr5[3]*/\
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vmla.f32 q14, q12, %f[wr3][1]\n" /*3579*wr3[3]*/\
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vadd.f32 q14, q14, q15\n"
#define RESULT_S2                                  \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vst1.f32 {d28-d29}, [%[dout_ptr]]!\n"           \
  "bne 1b"
#define RESULT_S2_RELU                             \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vmax.f32 q14, q14, %q[vzero]\n"                 \
  "vst1.f32 {d28-d29}, [%[dout_ptr]]!\n"           \
  "bne 1b"
#define RESULT_S2_RELU6                            \
  "vld1.f32 {d26-d27}, [%[six_ptr]]\n"             \
  "vmax.f32 q14, q14, %q[vzero]\n"                 \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vmin.f32 q14, q14, q13\n"                       \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vst1.f32 {d28-d29}, [%[dout_ptr]]!\n"           \
  "bne 1b"
#define RESULT_S2_LEAKY_RELU                       \
  "vld1.f32 {d26-d27}, [%[scale_ptr]]\n"           \
  "vcge.f32 q11, q14, %q[vzero]\n"                 \
  "vmul.f32 q12, q14, q13\n"                       \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vbif q14, q12, q11\n"                           \
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vst1.f32 {d28-d29}, [%[dout_ptr]]!\n"           \
  "bne 1b"
#define RESULT_S2_OUT2                             \
  "vst1.f32 {d30-d31}, [%[dout_ptr0]]!\n"          \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vst1.f32 {d28-d29}, [%[dout_ptr1]]!\n"          \
  "bne 1b"
#define RESULT_S2_RELU_OUT2                        \
  "vmax.f32 q15, q15, %q[vzero]\n"                 \
  "vmax.f32 q14, q14, %q[vzero]\n"                 \
  "vst1.f32 {d30-d31}, [%[dout_ptr0]]!\n"          \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vst1.f32 {d28-d29}, [%[dout_ptr1]]!\n"          \
  "bne 1b"
#define RESULT_S2_RELU6_OUT2                       \
  "vld1.f32 {d26-d27}, [%[six_ptr]]\n"             \
  "vmax.f32 q15, q15, %q[vzero]\n"                 \
  "vmax.f32 q14, q14, %q[vzero]\n"                 \
  "vmin.f32 q15, q15, q13\n"                       \
  "vmin.f32 q14, q14, q13\n"                       \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vst1.f32 {d30-d31}, [%[dout_ptr0]]!\n"          \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vst1.f32 {d28-d29}, [%[dout_ptr1]]!\n"          \
  "bne 1b"
#define RESULT_S2_LEAKY_RELU_OUT2                  \
  "vld1.f32 {d26-d27}, [%[scale_ptr]]\n"           \
  "vcge.f32 q11, q15, %q[vzero]\n"                 \
  "vmul.f32 q12, q15, q13\n"                       \
  "vbif q15, q12, q11\n"                           \
  "vcge.f32 q11, q14, %q[vzero]\n"                 \
  "vmul.f32 q12, q14, q13\n"                       \
  "vext.32 q13, q8, q10, #2\n"           /*46810*/ \
  "vst1.f32 {d30-d31}, [%[dout_ptr0]]!\n"          \
  "vbif q14, q12, q11\n"                           \
  "vext.32 q11, q8, q10, #1\n"           /*2468*/  \
  "vext.32 d25, d19, d21, #1\n"          /*57-79*/ \
  "vld1.f32 {d30-d31}, [%[bias]]\n"                \
  "vst1.f32 {d28-d29}, [%[dout_ptr1]]!\n"          \
  "bne 1b"

#endif
// clang-format on
inline float compute_one_data_pre(
    const float* data, float32x4_t wr, float bias_val, float wei_val, int num) {
  float sum = bias_val;
  int index = 4 - num;
  for (int i = 0; i < num; i++) {
    sum += data[i] * wr[index + i];
  }
  sum += data[num] * wei_val;
  return sum;
}

inline float compute_one_data_post(
    const float* data, float32x4_t wr, float bias_val, float wei_val, int num) {
  float sum = bias_val;
  for (int i = 0; i < num; i++) {
    sum += data[i] * wr[i];
  }
  sum += data[num] * wei_val;
  return sum;
}

inline void compute_all_padding_pre(float* dout,
                                    const float** din_ptr_arr,
                                    const float* bias,
                                    float32x4_t* weights,
                                    bool odds,
                                    int pad_left,
                                    int pad_right,
                                    int num_index_left,
                                    int num_index_right,
                                    int cnt,
                                    int remain,
                                    int num) {
  int tmp_index = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp_index - k],
                                  weights[3 - k],
                                  0.f,
                                  weights[5][3 - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp_index - k]++;
    }
  }
  // mid
  // clang-format off
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - i],
                                   weights[3 - i],
                                   0.f,
                                   weights[5][3 - i],
                                   4);
      din_ptr_arr[tmp_index - i] += 2;
    }
    *dout++ = sum;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[4],
                                      bias[0],
                                      weights[4][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - k],
                                   weights[3 - k],
                                   0.f,
                                   weights[3 - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp_index - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum;
  }
}
inline void compute_all_padding_mid(float* dout,
                                    const float** din_ptr_arr,
                                    const float* bias,
                                    float32x4_t* weights,
                                    bool odds,
                                    int pad_left,
                                    int pad_right,
                                    int num_index_left,
                                    int num_index_right,
                                    int cnt,
                                    int remain,
                                    int num) {
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum;
  }
}
inline void compute_all_padding_mid_out2(float* dout0,
                                         float* dout1,
                                         const float** din_ptr_arr,
                                         const float* bias,
                                         float32x4_t* weights,
                                         bool odds,
                                         int pad_left,
                                         int pad_right,
                                         int num_index_left,
                                         int num_index_right,
                                         int cnt,
                                         int remain,
                                         int num) {
  int tmp1 = num + 2;
  int tmp2 = num + 1;
  int tmp = num - 1;
  // left
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    float sum1 = compute_one_data_pre(din_ptr_arr[tmp1],
                                      weights[num],
                                      bias[0],
                                      weights[6][0],
                                      num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
      sum1 += compute_one_data_pre(din_ptr_arr[tmp2 - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[5][tmp - k],
                                   num_index_left);
    }
    num_index_left += 2;
    *dout0++ = sum;
    *dout1++ = sum1;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[tmp1]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp2 - k]++;
    }
    din_ptr_arr[1]++;
    din_ptr_arr[0]++;
  }
  // clang-format off
  // mid
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    float sum1 = compute_one_data_post(
        din_ptr_arr[tmp1], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[tmp1] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      sum1 += compute_one_data_post(
          din_ptr_arr[tmp2 - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp2 - i] += 2;
    }
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum;
    *dout1++ = sum1;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    float sum1 = compute_one_data_post(din_ptr_arr[tmp1],
                                       weights[num],
                                       bias[0],
                                       weights[num][num_index_right],
                                       num_index_right);
    din_ptr_arr[tmp1] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      sum1 += compute_one_data_post(din_ptr_arr[tmp2 - k],
                                    weights[tmp - k],
                                    0.f,
                                    weights[tmp - k][num_index_right],
                                    num_index_right);
      din_ptr_arr[tmp2 - k] += 2;
    }
    num_index_right -= 2;
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum;
    *dout1++ = sum1;
  }
}

inline void compute_all_padding_post(float* dout,
                                     const float** din_ptr_arr,
                                     const float* bias,
                                     float32x4_t* weights,
                                     bool odds,
                                     int pad_left,
                                     int pad_right,
                                     int num_index_left,
                                     int num_index_right,
                                     int cnt,
                                     int remain,
                                     int num) {
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(din_ptr_arr[num],
                                     weights[num],
                                     bias[0],
                                     weights[5][num],
                                     num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[num] -= 8;
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp] -= 8;
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp - 1] -= 8;
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[0] -= 8;
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[5][num], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum;
  }
}

void conv_depthwise_5x5s2_bias(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int pad_top,
                               int pad_bottom,
                               int pad_left,
                               int pad_right,
                               ARMContext* ctx) {
  int in_size = win * hin;
  int out_size = wout * hout;
  int in_channel_size = chin * in_size;
  int out_channel_size = chin * out_size;
  int pad_left_new = (pad_left + 1) / 2;
  int pad_top_new = (pad_top + 1) / 2;
  int weights_size = 25;
  int num_out = wout << 1;
  int loop_w = wout - pad_left_new;
  int loop_h = hout - pad_top_new;
  bool odds_w = pad_left % 2;
  bool odds_h = pad_top % 2;
  int n_top_h = 4 - pad_top;
  int n_left_w = 4 - pad_left;
  int n_right_w = 4;
  int n_bottom_h = 4;
  int in_w_cnt = (win - 4) >> 1;
  int in_h_cnt = (hin - 4) >> 1;
  int in_w_remain = win - (in_w_cnt << 1);
  int in_h_remain = hin - (in_h_cnt << 1);
  if (odds_w) {
    n_right_w = in_w_remain - 1;
  } else {
    if (in_w_remain == 5) {
      in_w_cnt++;
      n_right_w = 3;
    } else {
      n_right_w = in_w_remain;
    }
  }
  if (odds_h) {
    n_bottom_h = in_h_remain - 1;
  } else {
    if (in_h_remain == 5) {
      in_h_cnt++;
      n_bottom_h = 2;
    } else {
      n_bottom_h = in_h_remain;
    }
  }
  int pad_right_new = loop_w - in_w_cnt;
  int pad_bottom_new = loop_h - in_h_cnt;
  int cnt = in_w_cnt >> 2;
  int remain = in_w_cnt & 3;
  n_bottom_h--;
  n_right_w--;
  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * in_channel_size;
    float* dout_batch = dout + n * out_channel_size;
#pragma omp parallel for
    for (int c = 0; c < chin; c++) {
      const float* din_ch = din_batch + c * in_size;
      const float* weights_ch = weights + c * weights_size;
      float* dout_ch = dout_batch + c * out_size;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float* din_ptr0 = din_ch;
      const float* din_ptr1 = din_ptr0 + win;
      const float* din_ptr2 = din_ptr1 + win;
      const float* din_ptr3 = din_ptr2 + win;
      const float* din_ptr4 = din_ptr3 + win;
      const float* din_ptr5 = din_ptr4 + win;
      const float* din_ptr6 = din_ptr5 + win;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float* dout_ptr0 = dout_ch;
      float* dout_ptr1 = dout_ch;
      float32x4_t wr5;
      float32x4_t wr6;
      float32x4_t wr0 = vld1q_f32(weights_ch);
      float32x4_t wr1 = vld1q_f32(weights_ch + 5);
      float32x4_t wr2 = vld1q_f32(weights_ch + 10);
      float32x4_t wr3 = vld1q_f32(weights_ch + 15);
      float32x4_t wr4 = vld1q_f32(weights_ch + 20);
      wr5 = vsetq_lane_f32(weights_ch[4], wr5, 0);
      wr5 = vsetq_lane_f32(weights_ch[9], wr5, 1);
      wr5 = vsetq_lane_f32(weights_ch[14], wr5, 2);
      wr5 = vsetq_lane_f32(weights_ch[19], wr5, 3);
      wr6 = vsetq_lane_f32(weights_ch[24], wr6, 0);
      const float* din_ptr_arr[] = {
          din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5, din_ptr6};
      float32x4_t weights_vec[] = {wr0, wr1, wr2, wr3, wr4, wr5, wr6};
      // top_h
      int h_in_num = n_top_h;
      for (int h = pad_top_new; h > 0; h--) {
        compute_all_padding_pre(dout_ptr0,
                                din_ptr_arr,
                                vbias,
                                weights_vec,
                                odds_w,
                                pad_left_new,
                                pad_right_new,
                                n_left_w,
                                n_right_w,
                                cnt,
                                remain,
                                h_in_num);
        dout_ptr0 += wout;
        h_in_num += 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      if (odds_h) {
        din_ptr0 = din_ptr1;
        din_ptr1 = din_ptr2;
        din_ptr2 = din_ptr3;
        din_ptr3 = din_ptr4;
        din_ptr4 = din_ptr5;
        din_ptr5 = din_ptr6;
        din_ptr6 += win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      dout_ptr1 = dout_ptr0 + wout;
      // mid_h
      for (int h = 0; h < in_h_cnt - 1; h += 2) {
        compute_all_padding_mid_out2(dout_ptr0,
                                     dout_ptr1,
                                     din_ptr_arr,
                                     vbias,
                                     weights_vec,
                                     odds_w,
                                     pad_left_new,
                                     pad_right_new,
                                     n_left_w,
                                     n_right_w,
                                     cnt,
                                     remain,
                                     4);
        dout_ptr0 += num_out;
        dout_ptr1 += num_out;
        din_ptr0 = din_ptr4;
        din_ptr1 = din_ptr5;
        din_ptr2 = din_ptr6;
        din_ptr3 = din_ptr6 + win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr4 = din_ptr3 + win;
        din_ptr_arr[2] = din_ptr2;
        din_ptr5 = din_ptr4 + win;
        din_ptr_arr[3] = din_ptr3;
        din_ptr6 = din_ptr5 + win;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      if (in_h_cnt % 2 != 0) {
        compute_all_padding_mid(dout_ptr0,
                                din_ptr_arr,
                                vbias,
                                weights_vec,
                                odds_w,
                                pad_left_new,
                                pad_right_new,
                                n_left_w,
                                n_right_w,
                                cnt,
                                remain,
                                4);
        dout_ptr0 = dout_ptr1;
        din_ptr0 = din_ptr2;
        din_ptr1 = din_ptr3;
        din_ptr2 = din_ptr4;
        din_ptr3 = din_ptr5;
        din_ptr4 = din_ptr6;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      // bottom
      h_in_num = n_bottom_h;
      for (int h = 0; h < pad_bottom_new; h++) {
        compute_all_padding_post(dout_ptr0,
                                 din_ptr_arr,
                                 vbias,
                                 weights_vec,
                                 odds_w,
                                 pad_left_new,
                                 pad_right_new,
                                 n_left_w,
                                 n_right_w,
                                 cnt,
                                 remain,
                                 h_in_num);
        dout_ptr0 += wout;
        h_in_num -= 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
    }
  }
}

inline void compute_all_padding_pre_relu(float* dout,
                                         const float** din_ptr_arr,
                                         const float* bias,
                                         float32x4_t* weights,
                                         float32x4_t vzero,
                                         bool odds,
                                         int pad_left,
                                         int pad_right,
                                         int num_index_left,
                                         int num_index_right,
                                         int cnt,
                                         int remain,
                                         int num) {
  int tmp_index = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp_index - k],
                                  weights[3 - k],
                                  0.f,
                                  weights[5][3 - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp_index - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - i],
                                   weights[3 - i],
                                   0.f,
                                   weights[5][3 - i],
                                   4);
      din_ptr_arr[tmp_index - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[4],
                                      bias[0],
                                      weights[4][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - k],
                                   weights[3 - k],
                                   0.f,
                                   weights[3 - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp_index - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
}
inline void compute_all_padding_mid_relu(float* dout,
                                         const float** din_ptr_arr,
                                         const float* bias,
                                         float32x4_t* weights,
                                         float32x4_t vzero,
                                         bool odds,
                                         int pad_left,
                                         int pad_right,
                                         int num_index_left,
                                         int num_index_right,
                                         int cnt,
                                         int remain,
                                         int num) {
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_RELU
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_RELU
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
     din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
}
inline void compute_all_padding_mid_relu_out2(float* dout0,
                                              float* dout1,
                                              const float** din_ptr_arr,
                                              const float* bias,
                                              float32x4_t* weights,
                                              float32x4_t vzero,
                                              bool odds,
                                              int pad_left,
                                              int pad_right,
                                              int num_index_left,
                                              int num_index_right,
                                              int cnt,
                                              int remain,
                                              int num) {
  // left
  int tmp1 = num + 2;
  int tmp2 = num + 1;
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    float sum1 = compute_one_data_pre(din_ptr_arr[tmp1],
                                      weights[num],
                                      bias[0],
                                      weights[6][0],
                                      num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
      sum1 += compute_one_data_pre(din_ptr_arr[tmp2 - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[5][tmp - k],
                                   num_index_left);
    }
    num_index_left += 2;
    *dout0++ = sum > 0.f ? sum : 0.f;
    *dout1++ = sum1 > 0.f ? sum1 : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[tmp1]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp2 - k]++;
    }
    din_ptr_arr[1]++;
    din_ptr_arr[0]++;
  }
  // clang-format off
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_RELU_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_RELU_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
     din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    float sum1 = compute_one_data_post(
        din_ptr_arr[tmp1], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[tmp1] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      sum1 += compute_one_data_post(
          din_ptr_arr[tmp2 - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp2 - i] += 2;
    }
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? sum : 0.f;
    *dout1++ = sum1 > 0.f ? sum1 : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    float sum1 = compute_one_data_post(din_ptr_arr[tmp1],
                                       weights[num],
                                       bias[0],
                                       weights[num][num_index_right],
                                       num_index_right);
    din_ptr_arr[tmp1] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      sum1 += compute_one_data_post(din_ptr_arr[tmp2 - k],
                                    weights[tmp - k],
                                    0.f,
                                    weights[tmp - k][num_index_right],
                                    num_index_right);
      din_ptr_arr[tmp2 - k] += 2;
    }
    num_index_right -= 2;
    din_ptr_arr[0] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? sum : 0.f;
    *dout1++ = sum1 > 0.f ? sum1 : 0.f;
  }
}
inline void compute_all_padding_post_relu(float* dout,
                                          const float** din_ptr_arr,
                                          const float* bias,
                                          float32x4_t* weights,
                                          float32x4_t vzero,
                                          bool odds,
                                          int pad_left,
                                          int pad_right,
                                          int num_index_left,
                                          int num_index_right,
                                          int cnt,
                                          int remain,
                                          int num) {
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(din_ptr_arr[num],
                                     weights[num],
                                     bias[0],
                                     weights[5][num],
                                     num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[num] -= 8;
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp] -= 8;
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp - 1] -= 8;
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[0] -= 8;
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << " does not support";
    }
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[5][num], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : 0.f;
  }
}

void conv_depthwise_5x5s2_bias_relu(float* dout,
                                    const float* din,
                                    const float* weights,
                                    const float* bias,
                                    bool flag_bias,
                                    int num,
                                    int chin,
                                    int hin,
                                    int win,
                                    int hout,
                                    int wout,
                                    int pad_top,
                                    int pad_bottom,
                                    int pad_left,
                                    int pad_right,
                                    ARMContext* ctx) {
  int in_size = win * hin;
  int out_size = wout * hout;
  int pad_left_new = (pad_left + 1) / 2;
  int pad_top_new = (pad_top + 1) / 2;
  int in_channel_size = chin * in_size;
  int out_channel_size = chin * out_size;
  int weights_size = 25;
  int num_out = wout << 1;
  int loop_w = wout - pad_left_new;
  int loop_h = hout - pad_top_new;
  bool odds_w = pad_left % 2;
  bool odds_h = pad_top % 2;
  int n_top_h = 4 - pad_top;
  int n_left_w = 4 - pad_left;
  int n_right_w = 4;
  int n_bottom_h = 4;
  int in_w_cnt = (win - 4) >> 1;
  int in_h_cnt = (hin - 4) >> 1;
  int in_w_remain = win - (in_w_cnt << 1);
  int in_h_remain = hin - (in_h_cnt << 1);
  if (odds_w) {
    n_right_w = in_w_remain - 1;
  } else {
    if (in_w_remain == 5) {
      in_w_cnt++;
      n_right_w = 3;
    } else {
      n_right_w = in_w_remain;
    }
  }
  if (odds_h) {
    n_bottom_h = in_h_remain - 1;
  } else {
    if (in_h_remain == 5) {
      in_h_cnt++;
      n_bottom_h = 2;
    } else {
      n_bottom_h = in_h_remain;
    }
  }
  int pad_right_new = loop_w - in_w_cnt;
  int pad_bottom_new = loop_h - in_h_cnt;
  int cnt = in_w_cnt >> 2;
  int remain = in_w_cnt & 3;
  n_bottom_h--;
  n_right_w--;
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * in_channel_size;
    float* dout_batch = dout + n * out_channel_size;
#pragma omp parallel for
    for (int c = 0; c < chin; c++) {
      const float* din_ch = din_batch + c * in_size;
      const float* weights_ch = weights + c * weights_size;
      float* dout_ch = dout_batch + c * out_size;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float* din_ptr0 = din_ch;
      const float* din_ptr1 = din_ptr0 + win;
      const float* din_ptr2 = din_ptr1 + win;
      const float* din_ptr3 = din_ptr2 + win;
      const float* din_ptr4 = din_ptr3 + win;
      const float* din_ptr5 = din_ptr4 + win;
      const float* din_ptr6 = din_ptr5 + win;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float* dout_ptr0 = dout_ch;
      float* dout_ptr1 = dout_ch;
      float32x4_t wr5;
      float32x4_t wr6;
      float32x4_t wr0 = vld1q_f32(weights_ch);
      float32x4_t wr1 = vld1q_f32(weights_ch + 5);
      float32x4_t wr2 = vld1q_f32(weights_ch + 10);
      float32x4_t wr3 = vld1q_f32(weights_ch + 15);
      float32x4_t wr4 = vld1q_f32(weights_ch + 20);
      wr5 = vsetq_lane_f32(weights_ch[4], wr5, 0);
      wr5 = vsetq_lane_f32(weights_ch[9], wr5, 1);
      wr5 = vsetq_lane_f32(weights_ch[14], wr5, 2);
      wr5 = vsetq_lane_f32(weights_ch[19], wr5, 3);
      wr6 = vsetq_lane_f32(weights_ch[24], wr6, 0);
      const float* din_ptr_arr[] = {
          din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5, din_ptr6};
      float32x4_t weights_vec[] = {wr0, wr1, wr2, wr3, wr4, wr5, wr6};
      // top_h
      int h_in_num = n_top_h;
      for (int h = pad_top_new; h > 0; h--) {
        compute_all_padding_pre_relu(dout_ptr0,
                                     din_ptr_arr,
                                     vbias,
                                     weights_vec,
                                     vzero,
                                     odds_w,
                                     pad_left_new,
                                     pad_right_new,
                                     n_left_w,
                                     n_right_w,
                                     cnt,
                                     remain,
                                     h_in_num);
        dout_ptr0 += wout;
        h_in_num += 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      if (odds_h) {
        din_ptr0 = din_ptr1;
        din_ptr1 = din_ptr2;
        din_ptr2 = din_ptr3;
        din_ptr3 = din_ptr4;
        din_ptr4 = din_ptr5;
        din_ptr5 = din_ptr6;
        din_ptr6 += win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      dout_ptr1 = dout_ptr0 + wout;
      // mid_h
      for (int h = 0; h < in_h_cnt - 1; h += 2) {
        compute_all_padding_mid_relu_out2(dout_ptr0,
                                          dout_ptr1,
                                          din_ptr_arr,
                                          vbias,
                                          weights_vec,
                                          vzero,
                                          odds_w,
                                          pad_left_new,
                                          pad_right_new,
                                          n_left_w,
                                          n_right_w,
                                          cnt,
                                          remain,
                                          4);
        dout_ptr0 += num_out;
        dout_ptr1 += num_out;
        din_ptr0 = din_ptr4;
        din_ptr1 = din_ptr5;
        din_ptr2 = din_ptr6;
        din_ptr3 = din_ptr6 + win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr4 = din_ptr3 + win;
        din_ptr_arr[2] = din_ptr2;
        din_ptr5 = din_ptr4 + win;
        din_ptr_arr[3] = din_ptr3;
        din_ptr6 = din_ptr5 + win;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      if (in_h_cnt % 2 != 0) {
        compute_all_padding_mid_relu(dout_ptr0,
                                     din_ptr_arr,
                                     vbias,
                                     weights_vec,
                                     vzero,
                                     odds_w,
                                     pad_left_new,
                                     pad_right_new,
                                     n_left_w,
                                     n_right_w,
                                     cnt,
                                     remain,
                                     4);
        dout_ptr0 = dout_ptr1;
        din_ptr0 = din_ptr2;
        din_ptr1 = din_ptr3;
        din_ptr2 = din_ptr4;
        din_ptr3 = din_ptr5;
        din_ptr4 = din_ptr6;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      // bottom
      h_in_num = n_bottom_h;
      for (int h = 0; h < pad_bottom_new; h++) {
        compute_all_padding_post_relu(dout_ptr0,
                                      din_ptr_arr,
                                      vbias,
                                      weights_vec,
                                      vzero,
                                      odds_w,
                                      pad_left_new,
                                      pad_right_new,
                                      n_left_w,
                                      n_right_w,
                                      cnt,
                                      remain,
                                      h_in_num);
        dout_ptr0 += wout;
        h_in_num -= 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
    }
  }
}

inline void compute_all_padding_pre_relu6(float* dout,
                                          const float** din_ptr_arr,
                                          const float* bias,
                                          const float* six,
                                          float32x4_t* weights,
                                          float32x4_t vzero,
                                          bool odds,
                                          int pad_left,
                                          int pad_right,
                                          int num_index_left,
                                          int num_index_right,
                                          int cnt,
                                          int remain,
                                          int num) {
#ifdef __aarch64__
  float32x4_t vsix = vld1q_f32(six);
#endif
  int tmp_index = num - 1;
  // left
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp_index - k],
                                  weights[3 - k],
                                  0.f,
                                  weights[5][3 - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp_index - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - i],
                                   weights[3 - i],
                                   0.f,
                                   weights[5][3 - i],
                                   4);
      din_ptr_arr[tmp_index - i] += 2;
    }
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[4],
                                      bias[0],
                                      weights[4][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - k],
                                   weights[3 - k],
                                   0.f,
                                   weights[3 - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp_index - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
}
inline void compute_all_padding_mid_relu6(float* dout,
                                          const float** din_ptr_arr,
                                          const float* bias,
                                          const float* six,
                                          float32x4_t* weights,
                                          float32x4_t vzero,
                                          bool odds,
                                          int pad_left,
                                          int pad_right,
                                          int num_index_left,
                                          int num_index_right,
                                          int cnt,
                                          int remain,
                                          int num) {
#ifdef __aarch64__
  float32x4_t vsix = vld1q_f32(six);
#endif
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_RELU6
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [vsix] "w"(vsix),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_RELU6
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [six_ptr] "r"(six),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
}

inline void compute_all_padding_mid_relu6_out2(float* dout0,
                                               float* dout1,
                                               const float** din_ptr_arr,
                                               const float* bias,
                                               const float* six,
                                               float32x4_t* weights,
                                               float32x4_t vzero,
                                               bool odds,
                                               int pad_left,
                                               int pad_right,
                                               int num_index_left,
                                               int num_index_right,
                                               int cnt,
                                               int remain,
                                               int num) {
#ifdef __aarch64__
  float32x4_t vsix = vld1q_f32(six);
#endif
  // left
  int tmp1 = num + 2;
  int tmp2 = num + 1;
  int tmp = num - 1;
  // clang-format off
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    float sum1 = compute_one_data_pre(din_ptr_arr[tmp1],
                                      weights[num],
                                      bias[0],
                                      weights[6][0],
                                      num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
      sum1 += compute_one_data_pre(din_ptr_arr[tmp2 - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[5][tmp - k],
                                   num_index_left);
    }
    num_index_left += 2;
    *dout0++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
    *dout1++ = sum1 > 0.f ? (sum1 < six[0] ? sum1 : six[0]) : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[tmp1]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp2 - k]++;
    }
    din_ptr_arr[1]++;
    din_ptr_arr[0]++;
  }
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_RELU6_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [vsix] "w"(vsix),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_RELU6_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [six_ptr] "r"(six),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    float sum1 = compute_one_data_post(
        din_ptr_arr[tmp1], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[tmp1] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      sum1 += compute_one_data_post(
          din_ptr_arr[tmp2 - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp2 - i] += 2;
    }
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
    *dout1++ = sum1 > 0.f ? (sum1 < six[0] ? sum1 : six[0]) : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    float sum1 = compute_one_data_post(din_ptr_arr[tmp1],
                                       weights[num],
                                       bias[0],
                                       weights[num][num_index_right],
                                       num_index_right);
    din_ptr_arr[tmp1] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      sum1 += compute_one_data_post(din_ptr_arr[tmp2 - k],
                                    weights[tmp - k],
                                    0.f,
                                    weights[tmp - k][num_index_right],
                                    num_index_right);
      din_ptr_arr[tmp2 - k] += 2;
    }
    num_index_right -= 2;
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
    *dout1++ = sum1 > 0.f ? (sum1 < six[0] ? sum1 : six[0]) : 0.f;
  }
}
inline void compute_all_padding_post_relu6(float* dout,
                                           const float** din_ptr_arr,
                                           const float* bias,
                                           const float* six,
                                           float32x4_t* weights,
                                           float32x4_t vzero,
                                           bool odds,
                                           int pad_left,
                                           int pad_right,
                                           int num_index_left,
                                           int num_index_right,
                                           int cnt,
                                           int remain,
                                           int num) {
#ifdef __aarch64__
  float32x4_t vsix = vld1q_f32(six);
#endif
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(din_ptr_arr[num],
                                     weights[num],
                                     bias[0],
                                     weights[5][num],
                                     num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[num] -= 8;
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp] -= 8;
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp - 1] -= 8;
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vsix] "w"(vsix),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_RELU6
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [six_ptr] "r"(six),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[0] -= 8;
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[3], weights[num], bias[0], weights[5][num], 4);
    din_ptr_arr[3] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[2 - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[2 - i] += 2;
    }
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? (sum < six[0] ? sum : six[0]) : 0.f;
  }
}

void conv_depthwise_5x5s2_bias_relu6(float* dout,
                                     const float* din,
                                     const float* weights,
                                     const float* bias,
                                     const float* six,
                                     bool flag_bias,
                                     int num,
                                     int chin,
                                     int hin,
                                     int win,
                                     int hout,
                                     int wout,
                                     int pad_top,
                                     int pad_bottom,
                                     int pad_left,
                                     int pad_right,
                                     ARMContext* ctx) {
  int in_size = win * hin;
  int out_size = wout * hout;
  int pad_left_new = (pad_left + 1) / 2;
  int pad_top_new = (pad_top + 1) / 2;
  int in_channel_size = chin * in_size;
  int out_channel_size = chin * out_size;
  int weights_size = 25;
  int num_out = wout << 1;
  int loop_w = wout - pad_left_new;
  int loop_h = hout - pad_top_new;
  bool odds_w = pad_left % 2;
  bool odds_h = pad_top % 2;
  int n_top_h = 4 - pad_top;
  int n_left_w = 4 - pad_left;
  int n_right_w = 4;
  int n_bottom_h = 4;
  int in_w_cnt = (win - 4) >> 1;
  int in_h_cnt = (hin - 4) >> 1;
  int in_w_remain = win - (in_w_cnt << 1);
  int in_h_remain = hin - (in_h_cnt << 1);
  if (odds_w) {
    n_right_w = in_w_remain - 1;
  } else {
    if (in_w_remain == 5) {
      in_w_cnt++;
      n_right_w = 3;
    } else {
      n_right_w = in_w_remain;
    }
  }
  if (odds_h) {
    n_bottom_h = in_h_remain - 1;
  } else {
    if (in_h_remain == 5) {
      in_h_cnt++;
      n_bottom_h = 2;
    } else {
      n_bottom_h = in_h_remain;
    }
  }
  int pad_right_new = loop_w - in_w_cnt;
  int pad_bottom_new = loop_h - in_h_cnt;
  int cnt = in_w_cnt >> 2;
  int remain = in_w_cnt & 3;
  n_bottom_h--;
  n_right_w--;
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * in_channel_size;
    float* dout_batch = dout + n * out_channel_size;
#pragma omp parallel for
    for (int c = 0; c < chin; c++) {
      const float* din_ch = din_batch + c * in_size;
      const float* weights_ch = weights + c * weights_size;
      float* dout_ch = dout_batch + c * out_size;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float* din_ptr0 = din_ch;
      const float* din_ptr1 = din_ptr0 + win;
      const float* din_ptr2 = din_ptr1 + win;
      const float* din_ptr3 = din_ptr2 + win;
      const float* din_ptr4 = din_ptr3 + win;
      const float* din_ptr5 = din_ptr4 + win;
      const float* din_ptr6 = din_ptr5 + win;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float* dout_ptr0 = dout_ch;
      float* dout_ptr1 = dout_ch;
      float32x4_t wr5;
      float32x4_t wr6;
      float32x4_t wr0 = vld1q_f32(weights_ch);
      float32x4_t wr1 = vld1q_f32(weights_ch + 5);
      float32x4_t wr2 = vld1q_f32(weights_ch + 10);
      float32x4_t wr3 = vld1q_f32(weights_ch + 15);
      float32x4_t wr4 = vld1q_f32(weights_ch + 20);
      wr5 = vsetq_lane_f32(weights_ch[4], wr5, 0);
      wr5 = vsetq_lane_f32(weights_ch[9], wr5, 1);
      wr5 = vsetq_lane_f32(weights_ch[14], wr5, 2);
      wr5 = vsetq_lane_f32(weights_ch[19], wr5, 3);
      wr6 = vsetq_lane_f32(weights_ch[24], wr6, 0);
      const float* din_ptr_arr[] = {
          din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5, din_ptr6};
      float32x4_t weights_vec[] = {wr0, wr1, wr2, wr3, wr4, wr5, wr6};
      // top_h
      int h_in_num = n_top_h;
      for (int h = pad_top_new; h > 0; h--) {
        compute_all_padding_pre_relu6(dout_ptr0,
                                      din_ptr_arr,
                                      vbias,
                                      six,
                                      weights_vec,
                                      vzero,
                                      odds_w,
                                      pad_left_new,
                                      pad_right_new,
                                      n_left_w,
                                      n_right_w,
                                      cnt,
                                      remain,
                                      h_in_num);
        dout_ptr0 += wout;
        h_in_num += 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      if (odds_h) {
        din_ptr0 = din_ptr1;
        din_ptr1 = din_ptr2;
        din_ptr2 = din_ptr3;
        din_ptr3 = din_ptr4;
        din_ptr4 = din_ptr5;
        din_ptr5 = din_ptr6;
        din_ptr6 += win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      dout_ptr1 = dout_ptr0 + wout;
      // mid_h
      for (int h = 0; h < in_h_cnt - 1; h += 2) {
        compute_all_padding_mid_relu6_out2(dout_ptr0,
                                           dout_ptr1,
                                           din_ptr_arr,
                                           vbias,
                                           six,
                                           weights_vec,
                                           vzero,
                                           odds_w,
                                           pad_left_new,
                                           pad_right_new,
                                           n_left_w,
                                           n_right_w,
                                           cnt,
                                           remain,
                                           4);
        dout_ptr0 += num_out;
        dout_ptr1 += num_out;
        din_ptr0 = din_ptr4;
        din_ptr1 = din_ptr5;
        din_ptr2 = din_ptr6;
        din_ptr3 = din_ptr6 + win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr4 = din_ptr3 + win;
        din_ptr_arr[2] = din_ptr2;
        din_ptr5 = din_ptr4 + win;
        din_ptr_arr[3] = din_ptr3;
        din_ptr6 = din_ptr5 + win;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      if (in_h_cnt % 2 != 0) {
        compute_all_padding_mid_relu6(dout_ptr0,
                                      din_ptr_arr,
                                      vbias,
                                      six,
                                      weights_vec,
                                      vzero,
                                      odds_w,
                                      pad_left_new,
                                      pad_right_new,
                                      n_left_w,
                                      n_right_w,
                                      cnt,
                                      remain,
                                      4);
        dout_ptr0 = dout_ptr1;
        din_ptr0 = din_ptr2;
        din_ptr1 = din_ptr3;
        din_ptr2 = din_ptr4;
        din_ptr3 = din_ptr5;
        din_ptr4 = din_ptr6;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      // bottom
      h_in_num = n_bottom_h;
      for (int h = 0; h < pad_bottom_new; h++) {
        compute_all_padding_post_relu6(dout_ptr0,
                                       din_ptr_arr,
                                       vbias,
                                       six,
                                       weights_vec,
                                       vzero,
                                       odds_w,
                                       pad_left_new,
                                       pad_right_new,
                                       n_left_w,
                                       n_right_w,
                                       cnt,
                                       remain,
                                       h_in_num);
        dout_ptr0 += wout;
        h_in_num -= 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
    }
  }
}

inline void compute_all_padding_pre_leakyRelu(float* dout,
                                              const float** din_ptr_arr,
                                              const float* bias,
                                              const float* scale,
                                              float32x4_t* weights,
                                              float32x4_t vzero,
                                              bool odds,
                                              int pad_left,
                                              int pad_right,
                                              int num_index_left,
                                              int num_index_right,
                                              int cnt,
                                              int remain,
                                              int num) {
#ifdef __aarch64__
  float32x4_t vscale = vld1q_f32(scale);
#endif
  int tmp_index = num - 1;
  // left
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp_index - k],
                                  weights[3 - k],
                                  0.f,
                                  weights[5][3 - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp_index - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[4]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[3]),
                      [wr1] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[2]),
                      [wr1] "w"(weights[3]),
                      [wr2] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_PRE RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[1]),
                      [wr1] "w"(weights[2]),
                      [wr2] "w"(weights[3]),
                      [wr3] "w"(weights[4]),
                      [wr5] "w"(weights[5]),
                      [wr6] "w"(weights[6]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[4], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - i],
                                   weights[3 - i],
                                   0.f,
                                   weights[5][3 - i],
                                   4);
      din_ptr_arr[tmp_index - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[4],
                                      bias[0],
                                      weights[4][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp_index - k],
                                   weights[3 - k],
                                   0.f,
                                   weights[3 - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp_index - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
}
inline void compute_all_padding_mid_leakyRelu(float* dout,
                                              const float** din_ptr_arr,
                                              const float* bias,
                                              const float* scale,
                                              float32x4_t* weights,
                                              float32x4_t vzero,
                                              bool odds,
                                              int pad_left,
                                              int pad_right,
                                              int num_index_left,
                                              int num_index_right,
                                              int cnt,
                                              int remain,
                                              int num) {
#ifdef __aarch64__
  float32x4_t vscale = vld1q_f32(scale);
#endif
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_LEAKY_RELU
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [vscale] "w"(vscale),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                    "v17",
                    "v18");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2 RESULT_S2_LEAKY_RELU
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [dout_ptr] "+r"(dout)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [scale_ptr] "r"(scale),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
}
inline void compute_all_padding_mid_leakyRelu_out2(float* dout0,
                                                   float* dout1,
                                                   const float** din_ptr_arr,
                                                   const float* bias,
                                                   const float* scale,
                                                   float32x4_t* weights,
                                                   float32x4_t vzero,
                                                   bool odds,
                                                   int pad_left,
                                                   int pad_right,
                                                   int num_index_left,
                                                   int num_index_right,
                                                   int cnt,
                                                   int remain,
                                                   int num) {
#ifdef __aarch64__
  float32x4_t vscale = vld1q_f32(scale);
#endif
  // left
  int tmp1 = num + 2;
  int tmp2 = num + 1;
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], num_index_left);
    float sum1 = compute_one_data_pre(din_ptr_arr[tmp1],
                                      weights[num],
                                      bias[0],
                                      weights[6][0],
                                      num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
      sum1 += compute_one_data_pre(din_ptr_arr[tmp2 - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[5][tmp - k],
                                   num_index_left);
    }
    num_index_left += 2;
    *dout0++ = sum > 0.f ? sum : sum * scale[0];
    *dout1++ = sum1 > 0.f ? sum1 : sum1 * scale[0];
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[tmp1]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp2 - k]++;
    }
    din_ptr_arr[1]++;
    din_ptr_arr[0]++;
  }
  // clang-format off
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_LEAKY_RELU_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [vscale] "w"(vscale),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18",
                   "v19",
                   "v20",
                   "v21",
                   "v22");
#else
    asm volatile(COMPUTE_FIVE_LINE_S2_OUT2 RESULT_S2_LEAKY_RELU_OUT2
                 : [cnt] "+r"(cnt),
                   [din_ptr0] "+r"(din_ptr_arr[0]),
                   [din_ptr1] "+r"(din_ptr_arr[1]),
                   [din_ptr2] "+r"(din_ptr_arr[2]),
                   [din_ptr3] "+r"(din_ptr_arr[3]),
                   [din_ptr4] "+r"(din_ptr_arr[4]),
                   [din_ptr5] "+r"(din_ptr_arr[5]),
                   [din_ptr6] "+r"(din_ptr_arr[6]),
                   [dout_ptr0] "+r"(dout0),
                   [dout_ptr1] "+r"(dout1)
                 : [wr0] "w"(weights[0]),
                   [wr1] "w"(weights[1]),
                   [wr2] "w"(weights[2]),
                   [wr3] "w"(weights[3]),
                   [wr4] "w"(weights[4]),
                   [wr5] "w"(weights[5]),
                   [wr6] "w"(weights[6]),
                   [vzero] "w"(vzero),
                   [scale_ptr] "r"(scale),
                   [bias] "r"(bias)
                 : "cc",
                   "memory",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "q14",
                   "q15");
#endif
    din_ptr_arr[0] -= 8;
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[6][0], 4);
    float sum1 = compute_one_data_post(
        din_ptr_arr[tmp1], weights[num], bias[0], weights[6][0], 4);
    din_ptr_arr[tmp1] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      sum1 += compute_one_data_post(
          din_ptr_arr[tmp2 - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp2 - i] += 2;
    }
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? sum : sum * scale[0];
    *dout1++ = sum1 > 0.f ? sum1 : sum1 * scale[0];
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    float sum1 = compute_one_data_post(din_ptr_arr[tmp1],
                                       weights[num],
                                       bias[0],
                                       weights[num][num_index_right],
                                       num_index_right);
    din_ptr_arr[tmp1] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      sum1 += compute_one_data_post(din_ptr_arr[tmp2 - k],
                                    weights[tmp - k],
                                    0.f,
                                    weights[tmp - k][num_index_right],
                                    num_index_right);
      din_ptr_arr[tmp2 - k] += 2;
    }
    num_index_right -= 2;
    din_ptr_arr[1] += 2;
    din_ptr_arr[0] += 2;
    *dout0++ = sum > 0.f ? sum : sum * scale[0];
    *dout1++ = sum1 > 0.f ? sum1 : sum1 * scale[0];
  }
}
inline void compute_all_padding_post_leakyRelu(float* dout,
                                               const float** din_ptr_arr,
                                               const float* bias,
                                               const float* scale,
                                               float32x4_t* weights,
                                               float32x4_t vzero,
                                               bool odds,
                                               int pad_left,
                                               int pad_right,
                                               int num_index_left,
                                               int num_index_right,
                                               int cnt,
                                               int remain,
                                               int num) {
#ifdef __aarch64__
  float32x4_t vscale = vld1q_f32(scale);
#endif
  // left
  int tmp = num - 1;
  for (int i = pad_left; i > 0; i--) {
    float sum = compute_one_data_pre(din_ptr_arr[num],
                                     weights[num],
                                     bias[0],
                                     weights[5][num],
                                     num_index_left);
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_pre(din_ptr_arr[tmp - k],
                                  weights[tmp - k],
                                  0.f,
                                  weights[5][tmp - k],
                                  num_index_left);
    }
    num_index_left += 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  if (odds) {  // origin pad_left is odds, such as ori_pad_left=1
    din_ptr_arr[num]++;
    for (int k = 0; k < num; k++) {
      din_ptr_arr[tmp - k]++;
    }
  }
  // clang-format off
  // mid
  if (cnt > 0) {
    switch (num) {
      case 0:
#ifdef __aarch64__
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_ONE_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[num] -= 8;
        break;
      case 1:
#ifdef __aarch64__
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_TWO_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp]),
                      [din_ptr1] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp] -= 8;
        break;
      case 2:
#ifdef __aarch64__
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_THREE_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[tmp - 1]),
                      [din_ptr1] "+r"(din_ptr_arr[tmp]),
                      [din_ptr2] "+r"(din_ptr_arr[num]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[tmp - 1] -= 8;
        break;
      case 3:
#ifdef __aarch64__
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [vscale] "w"(vscale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "v9",
                      "v10",
                      "v11",
                      "v12",
                      "v13",
                      "v14",
                      "v15",
                      "v16",
                      "v17",
                      "v18");
#else
        asm volatile(COMPUTE_FOUR_LINE_S2_POST RESULT_S2_LEAKY_RELU
                    : [cnt] "+r"(cnt),
                      [din_ptr0] "+r"(din_ptr_arr[0]),
                      [din_ptr1] "+r"(din_ptr_arr[1]),
                      [din_ptr2] "+r"(din_ptr_arr[2]),
                      [din_ptr3] "+r"(din_ptr_arr[3]),
                      [dout_ptr] "+r"(dout)
                    : [wr0] "w"(weights[0]),
                      [wr1] "w"(weights[1]),
                      [wr2] "w"(weights[2]),
                      [wr3] "w"(weights[3]),
                      [wr5] "w"(weights[5]),
                      [vzero] "w"(vzero),
                      [scale_ptr] "r"(scale),
                      [bias] "r"(bias)
                    : "cc",
                      "memory",
                      "q8",
                      "q9",
                      "q10",
                      "q11",
                      "q12",
                      "q13",
                      "q14",
                      "q15");
#endif
        din_ptr_arr[0] -= 8;
        break;
      default:
        LOG(FATAL) << "This num: " << (num + 1) << "does not support";
    }
  }
  // clang-format on
  // remain
  for (int w = 0; w < remain; w++) {
    float sum = compute_one_data_post(
        din_ptr_arr[num], weights[num], bias[0], weights[5][num], 4);
    din_ptr_arr[num] += 2;
    for (int i = 0; i < num; i++) {
      sum += compute_one_data_post(
          din_ptr_arr[tmp - i], weights[tmp - i], 0.f, weights[5][tmp - i], 4);
      din_ptr_arr[tmp - i] += 2;
    }
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
  // right
  for (int i = 0; i < pad_right; i++) {
    float sum = compute_one_data_post(din_ptr_arr[num],
                                      weights[num],
                                      bias[0],
                                      weights[num][num_index_right],
                                      num_index_right);
    din_ptr_arr[num] += 2;
    for (int k = 0; k < num; k++) {
      sum += compute_one_data_post(din_ptr_arr[tmp - k],
                                   weights[tmp - k],
                                   0.f,
                                   weights[tmp - k][num_index_right],
                                   num_index_right);
      din_ptr_arr[tmp - k] += 2;
    }
    num_index_right -= 2;
    *dout++ = sum > 0.f ? sum : sum * scale[0];
  }
}

void conv_depthwise_5x5s2_bias_leakyRelu(float* dout,
                                         const float* din,
                                         const float* weights,
                                         const float* bias,
                                         const float* scale,
                                         bool flag_bias,
                                         int num,
                                         int chin,
                                         int hin,
                                         int win,
                                         int hout,
                                         int wout,
                                         int pad_top,
                                         int pad_bottom,
                                         int pad_left,
                                         int pad_right,
                                         ARMContext* ctx) {
  int in_size = win * hin;
  int out_size = wout * hout;
  int pad_left_new = (pad_left + 1) / 2;
  int pad_top_new = (pad_top + 1) / 2;
  int in_channel_size = chin * in_size;
  int out_channel_size = chin * out_size;
  int weights_size = 25;
  int num_out = wout << 1;
  int loop_w = wout - pad_left_new;
  int loop_h = hout - pad_top_new;
  bool odds_w = pad_left % 2;
  bool odds_h = pad_top % 2;
  int n_top_h = 4 - pad_top;
  int n_left_w = 4 - pad_left;
  int n_right_w = 4;
  int n_bottom_h = 4;
  int in_w_cnt = (win - 4) >> 1;
  int in_h_cnt = (hin - 4) >> 1;
  int in_w_remain = win - (in_w_cnt << 1);
  int in_h_remain = hin - (in_h_cnt << 1);
  if (odds_w) {
    n_right_w = in_w_remain - 1;
  } else {
    if (in_w_remain == 5) {
      in_w_cnt++;
      n_right_w = 3;
    } else {
      n_right_w = in_w_remain;
    }
  }
  if (odds_h) {
    n_bottom_h = in_h_remain - 1;
  } else {
    if (in_h_remain == 5) {
      in_h_cnt++;
      n_bottom_h = 2;
    } else {
      n_bottom_h = in_h_remain;
    }
  }
  int pad_right_new = loop_w - in_w_cnt;
  int pad_bottom_new = loop_h - in_h_cnt;
  int cnt = in_w_cnt >> 2;
  int remain = in_w_cnt & 3;
  n_bottom_h--;
  n_right_w--;
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * in_channel_size;
    float* dout_batch = dout + n * out_channel_size;
#pragma omp parallel for
    for (int c = 0; c < chin; c++) {
      const float* din_ch = din_batch + c * in_size;
      const float* weights_ch = weights + c * weights_size;
      float* dout_ch = dout_batch + c * out_size;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float* din_ptr0 = din_ch;
      const float* din_ptr1 = din_ptr0 + win;
      const float* din_ptr2 = din_ptr1 + win;
      const float* din_ptr3 = din_ptr2 + win;
      const float* din_ptr4 = din_ptr3 + win;
      const float* din_ptr5 = din_ptr4 + win;
      const float* din_ptr6 = din_ptr5 + win;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float* dout_ptr0 = dout_ch;
      float* dout_ptr1 = dout_ch;
      float32x4_t wr5;
      float32x4_t wr6;
      float32x4_t wr0 = vld1q_f32(weights_ch);
      float32x4_t wr1 = vld1q_f32(weights_ch + 5);
      float32x4_t wr2 = vld1q_f32(weights_ch + 10);
      float32x4_t wr3 = vld1q_f32(weights_ch + 15);
      float32x4_t wr4 = vld1q_f32(weights_ch + 20);
      wr5 = vsetq_lane_f32(weights_ch[4], wr5, 0);
      wr5 = vsetq_lane_f32(weights_ch[9], wr5, 1);
      wr5 = vsetq_lane_f32(weights_ch[14], wr5, 2);
      wr5 = vsetq_lane_f32(weights_ch[19], wr5, 3);
      wr6 = vsetq_lane_f32(weights_ch[24], wr6, 0);
      const float* din_ptr_arr[] = {
          din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5, din_ptr6};
      float32x4_t weights_vec[] = {wr0, wr1, wr2, wr3, wr4, wr5, wr6};
      // top_h
      int h_in_num = n_top_h;
      for (int h = pad_top_new; h > 0; h--) {
        compute_all_padding_pre_leakyRelu(dout_ptr0,
                                          din_ptr_arr,
                                          vbias,
                                          scale,
                                          weights_vec,
                                          vzero,
                                          odds_w,
                                          pad_left_new,
                                          pad_right_new,
                                          n_left_w,
                                          n_right_w,
                                          cnt,
                                          remain,
                                          h_in_num);
        dout_ptr0 += wout;
        h_in_num += 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      if (odds_h) {
        din_ptr0 = din_ptr1;
        din_ptr1 = din_ptr2;
        din_ptr2 = din_ptr3;
        din_ptr3 = din_ptr4;
        din_ptr4 = din_ptr5;
        din_ptr5 = din_ptr6;
        din_ptr6 += win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      dout_ptr1 = dout_ptr0 + wout;
      // mid_h
      for (int h = 0; h < in_h_cnt - 1; h += 2) {
        compute_all_padding_mid_leakyRelu_out2(dout_ptr0,
                                               dout_ptr1,
                                               din_ptr_arr,
                                               vbias,
                                               scale,
                                               weights_vec,
                                               vzero,
                                               odds_w,
                                               pad_left_new,
                                               pad_right_new,
                                               n_left_w,
                                               n_right_w,
                                               cnt,
                                               remain,
                                               4);
        dout_ptr0 += num_out;
        dout_ptr1 += num_out;
        din_ptr0 = din_ptr4;
        din_ptr1 = din_ptr5;
        din_ptr2 = din_ptr6;
        din_ptr3 = din_ptr6 + win;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr4 = din_ptr3 + win;
        din_ptr_arr[2] = din_ptr2;
        din_ptr5 = din_ptr4 + win;
        din_ptr_arr[3] = din_ptr3;
        din_ptr6 = din_ptr5 + win;
        din_ptr_arr[4] = din_ptr4;
        din_ptr_arr[5] = din_ptr5;
        din_ptr_arr[6] = din_ptr6;
      }
      if (in_h_cnt % 2 != 0) {
        compute_all_padding_mid_leakyRelu(dout_ptr0,
                                          din_ptr_arr,
                                          vbias,
                                          scale,
                                          weights_vec,
                                          vzero,
                                          odds_w,
                                          pad_left_new,
                                          pad_right_new,
                                          n_left_w,
                                          n_right_w,
                                          cnt,
                                          remain,
                                          4);
        dout_ptr0 = dout_ptr1;
        din_ptr0 = din_ptr2;
        din_ptr1 = din_ptr3;
        din_ptr2 = din_ptr4;
        din_ptr3 = din_ptr5;
        din_ptr4 = din_ptr6;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
      // bottom
      h_in_num = n_bottom_h;
      for (int h = 0; h < pad_bottom_new; h++) {
        compute_all_padding_post_leakyRelu(dout_ptr0,
                                           din_ptr_arr,
                                           vbias,
                                           scale,
                                           weights_vec,
                                           vzero,
                                           odds_w,
                                           pad_left_new,
                                           pad_right_new,
                                           n_left_w,
                                           n_right_w,
                                           cnt,
                                           remain,
                                           h_in_num);
        dout_ptr0 += wout;
        h_in_num -= 2;
        din_ptr_arr[0] = din_ptr0;
        din_ptr_arr[1] = din_ptr1;
        din_ptr_arr[2] = din_ptr2;
        din_ptr_arr[3] = din_ptr3;
        din_ptr_arr[4] = din_ptr4;
      }
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
