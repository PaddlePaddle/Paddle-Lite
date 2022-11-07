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

#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/backends/arm/math/fp16/conv3x3_depthwise_fp16.h"
#include "lite/backends/arm/math/fp16/conv_depthwise_common_fp16.h"
#include "lite/backends/arm/math/fp16/gemm_fp16.h"
#include "lite/backends/arm/math/fp16/gemv_fp16.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
/**
 * \brief inline funcs used in im2col
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/**
 * \brief normal im2col function for gemm conv
 * @tparam dtype
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
void im2col_common_fp16(IM2COL_PARAM(float16_t), int stride_h, int stride_w) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_top + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_left + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void im2col_s1_fp16(IM2COL_PARAM(float16_t)) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float16_t));

  LITE_PARALLEL_BEGIN(c, tid, channels) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * out_channel_size * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * out_channel_size;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset)), 0);
        int oh_end = std::min(((height + pad_bottom - h_offset)), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset)), 0);
        int ow_end = std::min(((width + pad_right - w_offset)), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
          int iw = ow_begin - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const float16_t* data_im_ptr = data_im + data_im_offset;
          float16_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 7 < ow_end; ow += 8, iw += 8) {
            float16x8_t tmp = vld1q_f16(data_im_ptr + iw);
            vst1q_f16(data_col_ptr + ow, tmp);
          }
          if (ow + 3 < ow_end) {
            float16x4_t tmp = vld1_f16(data_im_ptr + iw);
            vst1_f16(data_col_ptr + ow, tmp);
            ow += 4;
            iw += 4;
          }
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END()
}

void im2col_s2_fp16(IM2COL_PARAM(float16_t)) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float16_t));

  LITE_PARALLEL_BEGIN(c, tid, channels) {
    int data_im_z = c * in_channel_size;
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * output_h * output_w * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * output_h * output_w;
        int data_col_z = data_col_z1 + data_col_z2 + data_col_z3;
        int oh_begin = std::max(((pad_top - h_offset + 1) / 2), 0);
        int oh_end =
            std::min(((height + pad_bottom - h_offset + 1) / 2), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset + 1) / 2), 0);
        int ow_end =
            std::min(((width + pad_right - w_offset + 1) / 2), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin * 2 - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
          int iw = ow_begin * 2 - pad_left + w_offset;
          int ow = ow_begin;
          int data_im_offset = data_im_z + ih * width;
          int data_col_offset = data_col_z + oh * output_w;
          const float16_t* data_im_ptr = data_im + data_im_offset;
          float16_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 7 < ow_end; ow += 8, iw += 16) {
            float16x8x2_t tmp = vld2q_f16(data_im_ptr + iw);
            vst1q_f16(data_col_ptr + ow, tmp.val[0]);
          }
          if (ow + 3 < ow_end) {
            float16x4x2_t tmp = vld2_f16(data_im_ptr + iw);
            vst1_f16(data_col_ptr + ow, tmp.val[0]);
            ow += 4;
            iw += 8;
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END()
}
/**
 * \brief normal im2col function for gemm conv
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
void im2col_fp16(IM2COL_PARAM(float16_t), int stride_h, int stride_w) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
#define IM2COL_IN_PARAMS                                                     \
  data_im, channels, height, width, kernel_h, kernel_w, pad_top, pad_bottom, \
      pad_left, pad_right, dilation_h, dilation_w, data_col
  if (kspd && stride_h == 1) {
    im2col_s1_fp16(IM2COL_IN_PARAMS);
  } else if (kspd && stride_h == 2) {
    im2col_s2_fp16(IM2COL_IN_PARAMS);
  } else {
    im2col_common_fp16(IM2COL_IN_PARAMS, stride_h, stride_w);
  }
}

template <>
void col2im<float16_t>(const float16_t* data_col,
                       const int channels,
                       const int height,
                       const int width,
                       const int kernel_h,
                       const int kernel_w,
                       const int pad_h0,
                       const int pad_h1,
                       const int pad_w0,
                       const int pad_w1,
                       const int stride_h,
                       const int stride_w,
                       const int dilation_h,
                       const int dilation_w,
                       float16_t* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float16_t));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h0 + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w0 + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm
 * implementation
 */
void conv1x1s1_gemm_fp16(CONV_PARAM(float16_t)) {
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock_fp16(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float16_t* dout_group =
          static_cast<float16_t*>(o_data) + (b * oc + g * m) * channel_size_out;
      const float16_t* din_group = static_cast<const float16_t*>(i_data) +
                                   (b * ic + g * k) * channel_size_in;
      const float16_t* weights_group =
          static_cast<const float16_t*>(weights) + g * weights_size_per_group;
      const float16_t* bias_group = static_cast<const float16_t*>(bias) + g * m;
      if (n == 1) {
        gemv_fp16(weights_group,
                  din_group,
                  dout_group,
                  false,
                  m,
                  k,
                  0.f,
                  flag_bias,
                  bias_group,
                  act_param.has_active,
                  act_param,
                  ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float16_t* bias_ptr = new float16_t[n];
#else
        float16_t bias_ptr[n];  // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }

        gemv_fp16(din_group,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  0.f,
                  flag_bias,
                  bias_ptr,
                  act_param.has_active,
                  act_param,
                  ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
#endif
      } else {
        gemm_prepack_fp16(false,
                          m,
                          n,
                          k,
                          weights_group,
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
}

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm
 * implementation
 */
void conv_im2col_gemm_fp16(CONV_PARAM(float16_t)) {
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
  int hblock = get_hblock_fp16(ctx);
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
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float16_t* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const float16_t* din_group =
          i_data + (b * ic + g * chin_per_group) * channel_size_in;
      const float16_t* weights_group = weights + g * weights_size_per_group;
      const float16_t* bias_group = bias + g * m;
      float16_t* dB = tmp_work_space;
      im2col_fp16(din_group,
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
      if (n == 1) {
        gemv_fp16(weights_group,
                  dB,
                  dout_group,
                  false,
                  m,
                  k,
                  0.f,
                  flag_bias,
                  bias_group,
                  act_param.has_active,
                  act_param,
                  ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float16_t* bias_ptr = new float16_t[n];
#else
        float16_t bias_ptr[n];  // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }

        gemv_fp16(dB,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  0.f,
                  flag_bias,
                  bias_ptr,
                  act_param.has_active,
                  act_param,
                  ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
#endif
      } else {
        gemm_prepack_fp16(false,
                          m,
                          n,
                          k,
                          weights_group,
                          dB,
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
}

void conv_depthwise_3x3_fp16(CONV_PARAM(float16_t)) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  bool has_active = act_param.has_active;
  float16_t relu6_coff = static_cast<float16_t>(act_param.Relu_clipped_coef);
  float16_t leaky_alpha = static_cast<float16_t>(act_param.Leaky_relu_alpha);
  float16_t scale[8];
  switch (act_type) {
    case lite_api::ActivationType::kRelu6:
      for (int i = 0; i < 8; ++i) {
        scale[i] = relu6_coff;
      }
      break;
    case lite_api::ActivationType::kLeakyRelu:
      for (int i = 0; i < 8; ++i) {
        scale[i] = leaky_alpha;
      }
      break;
    default:
      break;
  }
#define CONV_DEPTHWISE_IN_PARAMS \
  o_data, i_data, weights, bias, scale, flag_bias, num, ic, ih, win, oh, ow, ctx
  if (has_active) {
    switch (act_type) {
      case lite_api::ActivationType::kRelu:
        if (stride == 1 && pad_h == 1 && pad_w == 1) {
          if (ow <= 8)
            conv_depthwise_3x3s1p1_bias_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p1_bias_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 1 && pad_h == 0 && pad_w == 0) {
          if (ow <= 8)
            conv_depthwise_3x3s1p0_bias_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p0_bias_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 1 && pad_w == 1) {
          if (win <= 15)
            conv_depthwise_3x3s2p1_bias_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p1_bias_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 0 && pad_w == 0) {
          if (win <= 16)
            conv_depthwise_3x3s2p0_bias_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p0_bias_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else {
          LOG(FATAL) << "fp16 3x3 depthwise conv fuse relu stride " << stride
                     << " pad_h " << pad_h << " pad_w " << pad_w
                     << "is not supported!";
        }
        break;
      case lite_api::ActivationType::kRelu6:
        if (stride == 1 && pad_h == 1 && pad_w == 1) {
          if (ow <= 8)
            conv_depthwise_3x3s1p1_bias_relu6_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p1_bias_relu6_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 1 && pad_h == 0 && pad_w == 0) {
          if (ow <= 8)
            conv_depthwise_3x3s1p0_bias_relu6_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p0_bias_relu6_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 1 && pad_w == 1) {
          if (win <= 15)
            conv_depthwise_3x3s2p1_bias_relu6_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p1_bias_relu6_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 0 && pad_w == 0) {
          if (win <= 16)
            conv_depthwise_3x3s2p0_bias_relu6_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p0_bias_relu6_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else {
          LOG(FATAL) << "fp16 3x3 depthwise conv fuse relu6 stride " << stride
                     << " pad_h " << pad_h << " pad_w " << pad_w
                     << "is not supported!";
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        if (stride == 1 && pad_h == 1 && pad_w == 1) {
          if (ow <= 8)
            conv_depthwise_3x3s1p1_bias_leaky_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p1_bias_leaky_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 1 && pad_h == 0 && pad_w == 0) {
          if (ow <= 8)
            conv_depthwise_3x3s1p0_bias_leaky_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s1p0_bias_leaky_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 1 && pad_w == 1) {
          if (win <= 15)
            conv_depthwise_3x3s2p1_bias_leaky_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p1_bias_leaky_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else if (stride == 2 && pad_h == 0 && pad_w == 0) {
          if (win <= 16)
            conv_depthwise_3x3s2p0_bias_leaky_relu_small_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
          else
            conv_depthwise_3x3s2p0_bias_leaky_relu_common_fp16_fp16(
                CONV_DEPTHWISE_IN_PARAMS);
        } else {
          LOG(FATAL) << "fp16 3x3 depthwise conv fuse leaky relu stride "
                     << stride << " pad_h " << pad_h << " pad_w " << pad_w
                     << "is not supported!";
        }
        break;
      default:
        LOG(FATAL) << "fp16 3x3 depthwise conv act_type: "
                   << static_cast<int>(act_type) << "is not supported!";
        break;
    }
  } else {
    if (stride == 1 && pad_h == 1 && pad_w == 1) {
      if (ow <= 8)
        conv_depthwise_3x3s1p1_bias_noact_small_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
      else
        conv_depthwise_3x3s1p1_bias_noact_common_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
    } else if (stride == 1 && pad_h == 0 && pad_w == 0) {
      if (ow <= 8)
        conv_depthwise_3x3s1p0_bias_noact_small_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
      else
        conv_depthwise_3x3s1p0_bias_noact_common_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
    } else if (stride == 2 && pad_h == 1 && pad_w == 1) {
      if (win <= 15)
        conv_depthwise_3x3s2p1_bias_noact_small_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
      else
        conv_depthwise_3x3s2p1_bias_noact_common_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
    } else if (stride == 2 && pad_h == 0 && pad_w == 0) {
      if (win <= 16)
        conv_depthwise_3x3s2p0_bias_noact_small_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
      else
        conv_depthwise_3x3s2p0_bias_noact_common_fp16_fp16(
            CONV_DEPTHWISE_IN_PARAMS);
    } else {
      LOG(FATAL) << "fp16 3x3 depthwise conv stride " << stride << "and pad_h "
                 << pad_h << "pad_w " << pad_w << "is not supported!";
    }
  }
}

void conv_depthwise_common(const float16_t* w_data,
                           const operators::ConvParam& param,
                           ARMContext* ctx) {
  const auto* i_data = param.x->data<float16_t>();
  const auto* b_data = param.bias ? param.bias->data<float16_t>() : nullptr;
  auto* o_data = param.output->mutable_data<float16_t>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int kh = w_dims[2];
  int kw = w_dims[3];
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];

  conv_depthwise_common_line(i_data,
                             o_data,
                             ic,
                             ih,
                             iw,
                             bs,
                             oc,
                             oh,
                             ow,
                             kh,
                             kw,
                             param.strides,
                             *param.dilations.get(),
                             *param.paddings.get(),
                             w_data,
                             b_data,
                             param,
                             ctx);
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
