// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/sve/conv_impl_sve.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/sve/gemm_sve.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm
 * implementation
*/
template <typename Dtype>
void conv1x1s1_gemm_sve(const Dtype* i_data,
                        Dtype* o_data,
                        int num,
                        int oc,
                        int oh,
                        int ow,
                        int ic,
                        int ih,
                        int win,
                        const Dtype* weights,
                        const Dtype* bias,
                        const operators::ConvParam& param,
                        ARMContext* ctx) {
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock_sve(ctx, m, sizeof(Dtype));
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      Dtype* dout_group =
          static_cast<Dtype*>(o_data) + (b * oc + g * m) * channel_size_out;
      const Dtype* din_group = static_cast<const Dtype*>(i_data) +
                               (b * ic + g * k) * channel_size_in;
      const Dtype* weights_group =
          static_cast<const Dtype*>(weights) + g * weights_size_per_group;
      const Dtype* bias_group = static_cast<const Dtype*>(bias) + g * m;

      sgemm_prepack_sve<Dtype>(false,
                               m,
                               n,
                               k,
                               weights_group,
                               k,
                               din_group,
                               n,
                               0.f,
                               dout_group,
                               n,
                               bias_group,
                               flag_bias,
                               act_param,
                               ctx);
    }
  }
}

template void conv1x1s1_gemm_sve<float>(const float* i_data,
                                        float* o_data,
                                        int num,
                                        int oc,
                                        int oh,
                                        int ow,
                                        int ic,
                                        int ih,
                                        int win,
                                        const float* weights,
                                        const float* bias,
                                        const operators::ConvParam& param,
                                        ARMContext* ctx);

#ifdef ENABLE_ARM_FP16
template void conv1x1s1_gemm_sve<float16_t>(const float16_t* i_data,
                                            float16_t* o_data,
                                            int num,
                                            int oc,
                                            int oh,
                                            int ow,
                                            int ic,
                                            int ih,
                                            int win,
                                            const float16_t* weights,
                                            const float16_t* bias,
                                            const operators::ConvParam& param,
                                            ARMContext* ctx);
#endif

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm
 * implementation
 */
template <>
void conv_im2col_gemm_sve(const float* i_data,
                          float* o_data,
                          int num,
                          int oc,
                          int oh,
                          int ow,
                          int ic,
                          int ih,
                          int win,
                          const float* weights,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx) {
  const int group = param.groups;
  auto filter_dims = param.filter->dims();
  const int kernel_h = filter_dims[2];
  const int kernel_w = filter_dims[3];  // nchw
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  int hblock = get_hblock_sve(ctx, m, 4);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;

  auto act_param = param.activation_param;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const float* din_group =
          i_data + (b * ic + g * chin_per_group) * channel_size_in;
      const float* weights_group = weights + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      float* dB = tmp_work_space;
      im2col<float>(din_group,
                    chin_per_group,
                    ih,
                    win,
                    kernel_h,
                    kernel_w,
                    paddings[0],
                    paddings[1],
                    paddings[2],
                    paddings[3],
                    param.strides[0],
                    param.strides[1],
                    dilations[0],
                    dilations[1],
                    dB);
      int ldb = n;
      sgemm_prepack_sve<float>(false,
                               m,
                               n,
                               k,
                               weights_group,
                               k,
                               dB,
                               ldb,
                               0.f,
                               dout_group,
                               n,
                               bias_group,
                               flag_bias,
                               act_param,
                               ctx);
    }
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void conv_im2col_gemm_sve(const float16_t* i_data,
                          float16_t* o_data,
                          int num,
                          int oc,
                          int oh,
                          int ow,
                          int ic,
                          int ih,
                          int win,
                          const float16_t* weights,
                          const float16_t* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx) {
  const int group = param.groups;
  auto filter_dims = param.filter->dims();
  const int kernel_h = filter_dims[2];
  const int kernel_w = filter_dims[3];  // nchw
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  int hblock = get_hblock_sve(ctx, m, 2);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;

  auto act_param = param.activation_param;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  float16_t* tmp_work_space =
      ctx->workspace_data<float16_t>() + ctx->llc_size() / sizeof(float16_t);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float16_t* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const float16_t* din_group =
          i_data + (b * ic + g * chin_per_group) * channel_size_in;
      const float16_t* weights_group = weights + g * weights_size_per_group;
      const float16_t* bias_group = bias + g * m;
      float16_t* dB = tmp_work_space;
      fp16::im2col_fp16(din_group,
                        chin_per_group,
                        ih,
                        win,
                        kernel_h,
                        kernel_w,
                        paddings[0],
                        paddings[1],
                        paddings[2],
                        paddings[3],
                        dilations[0],
                        dilations[1],
                        dB,
                        param.strides[0],
                        param.strides[1]);
      int ldb = n;
      sgemm_prepack_sve<float16_t>(false,
                                   m,
                                   n,
                                   k,
                                   weights_group,
                                   k,
                                   dB,
                                   ldb,
                                   0.f,
                                   dout_group,
                                   n,
                                   bias_group,
                                   flag_bias,
                                   act_param,
                                   ctx);
    }
  }
}
#endif

}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
