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

#include "lite/backends/arm/math/conv_impl.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/backends/arm/math/sgemv.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
#include "lite/backends/arm/math/sve/gemm_sve_i8mm.h"
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

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
template <typename Dtype>
void im2col_common(const Dtype* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   Dtype* data_col) {
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

template <>
void im2col_s1<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float));
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
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
          for (; ow + 3 < ow_end; ow += 4, iw += 4) {
            float32x4_t tmp = vld1q_f32(data_im_ptr + iw);
            vst1q_f32(data_col_ptr + ow, tmp);
          }
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

template <>
void im2col_s1<int8_t>(const int8_t* data_im,
                       int channels,
                       int height,
                       int width,
                       int kernel_h,
                       int kernel_w,
                       int pad_top,
                       int pad_bottom,
                       int pad_left,
                       int pad_right,
                       int dilation_h,
                       int dilation_w,
                       int8_t* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(int8_t));
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
          const int8_t* data_im_ptr = data_im + data_im_offset;
          int8_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 15 < ow_end; ow += 16, iw += 16) {
            int8x16_t tmp = vld1q_s8(data_im_ptr + iw);
            vst1q_s8(data_col_ptr + ow, tmp);
          }
          for (; ow + 7 < ow_end; ow += 8, iw += 8) {
            int8x8_t tmp = vld1_s8(data_im_ptr + iw);
            vst1_s8(data_col_ptr + ow, tmp);
          }
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

template <>
void im2col_s2<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(float));
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
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
          for (; ow + 3 < ow_end; ow += 4, iw += 8) {
            float32x4x2_t tmp = vld2q_f32(data_im_ptr + iw);
            vst1q_f32(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

template <>
void im2col_s2<int8_t>(const int8_t* data_im,
                       int channels,
                       int height,
                       int width,
                       int kernel_h,
                       int kernel_w,
                       int pad_top,
                       int pad_bottom,
                       int pad_left,
                       int pad_right,
                       int dilation_h,
                       int dilation_w,
                       int8_t* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
  memset(data_col, 0, output_plane_size * channels * sizeof(int8_t));
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
          const int8_t* data_im_ptr = data_im + data_im_offset;
          int8_t* data_col_ptr = data_col + data_col_offset;
          for (; ow + 15 < ow_end; ow += 16, iw += 32) {
            int8x16x2_t tmp = vld2q_s8(data_im_ptr + iw);
            vst1q_s8(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow + 7 < ow_end; ow += 8, iw += 16) {
            int8x8x2_t tmp = vld2_s8(data_im_ptr + iw);
            vst1_s8(data_col_ptr + ow, tmp.val[0]);
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
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
template <>
void im2col<float>(const float* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   float* data_col) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
  if (kspd && stride_h == 1) {
    im2col_s1<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else if (kspd && stride_h == 2) {
    im2col_s2<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else {
    im2col_common<float>(data_im,
                         channels,
                         height,
                         width,
                         kernel_h,
                         kernel_w,
                         pad_top,
                         pad_bottom,
                         pad_left,
                         pad_right,
                         stride_h,
                         stride_w,
                         dilation_h,
                         dilation_w,
                         data_col);
  }
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
template <>
void im2col<int8_t>(const int8_t* data_im,
                    int channels,
                    int height,
                    int width,
                    int kernel_h,
                    int kernel_w,
                    int pad_top,
                    int pad_bottom,
                    int pad_left,
                    int pad_right,
                    int stride_h,
                    int stride_w,
                    int dilation_h,
                    int dilation_w,
                    int8_t* data_col) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
  if (kspd && stride_h == 1) {
    im2col_s1<int8_t>(data_im,
                      channels,
                      height,
                      width,
                      kernel_h,
                      kernel_w,
                      pad_top,
                      pad_bottom,
                      pad_left,
                      pad_right,
                      dilation_h,
                      dilation_w,
                      data_col);
  } else if (kspd && stride_h == 2) {
    im2col_s2<int8_t>(data_im,
                      channels,
                      height,
                      width,
                      kernel_h,
                      kernel_w,
                      pad_top,
                      pad_bottom,
                      pad_left,
                      pad_right,
                      dilation_h,
                      dilation_w,
                      data_col);
  } else {
    im2col_common<int8_t>(data_im,
                          channels,
                          height,
                          width,
                          kernel_h,
                          kernel_w,
                          pad_top,
                          pad_bottom,
                          pad_left,
                          pad_right,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          data_col);
  }
}

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm
 * implementation
 */
void conv1x1s1_gemm(const float* i_data,
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
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock(ctx, m);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group =
          static_cast<float*>(o_data) + (b * oc + g * m) * channel_size_out;
      const float* din_group = static_cast<const float*>(i_data) +
                               (b * ic + g * k) * channel_size_in;
      const float* weights_group =
          static_cast<const float*>(weights) + g * weights_size_per_group;
      const float* bias_group = static_cast<const float*>(bias) + g * m;

      if (n == 1) {
        sgemv(weights_group,
              din_group,
              dout_group,
              false,
              m,
              k,
              0.f,
              flag_bias,
              bias_group,
              act_param,
              ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float* bias_ptr = new float[n];
#else
        float bias_ptr[n];   // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }

        sgemv(din_group,
              weights_group,
              dout_group,
              true,
              n,
              k,
              0.f,
              flag_bias,
              bias_ptr,
              act_param,
              ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
#endif
      } else {
        sgemm_prepack(false,
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

template <typename Dtype>
void conv1x1s1_gemm_int8(const int8_t* i_data,
                         Dtype* o_data,
                         int num,
                         int oc,
                         int oh,
                         int ow,
                         int ic,
                         int ih,
                         int win,
                         const int8_t* weights,
                         const float* bias,
                         const operators::ConvParam& param,
                         ARMContext* ctx,
                         const float* scale) {
  int group = param.groups;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;
  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
  if (ctx->has_sve2_i8mm()) {
    int hblock_sve = sve::get_hblock_int8_sve(ctx);
    int k_roundup_sve = ROUNDUP(k, sve::KBLOCK_INT8_SVE);
    int m_roundup_sve = ROUNDUP(m, hblock_sve);
    if (n > 1 && m > 1) {
      weights_size_per_group = ((m_roundup_sve * k_roundup_sve + 15) / 16) * 16;
    }
  } else {
#endif
    if (n > 1 && m > 1) {
      weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
    }
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
  }
#endif

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      Dtype* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const int8_t* din_group = i_data + (b * ic + g * k) * channel_size_in;
      const int8_t* weights_group = weights + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      const float* scale_group = scale + g * m;
      if (n == 1) {
        gemv_int8(weights_group,
                  din_group,
                  dout_group,
                  false,
                  m,
                  k,
                  scale_group,
                  flag_bias,
                  bias_group,
                  act_param,
                  ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float* bias_ptr = new float[n];
        float* scale_ptr = new float[n];
#else
        float bias_ptr[n];   // NOLINT
        float scale_ptr[n];  // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        for (int i = 0; i < n; i++) {
          scale_ptr[i] = scale_group[0];
        }
        gemv_int8(din_group,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  scale_ptr,
                  flag_bias,
                  bias_ptr,
                  act_param,
                  ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
        delete[] scale_ptr;
#endif
      } else {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
        if (ctx->has_sve2_i8mm()) {
          sve::gemm_prepack_int8_sve(weights_group,
                                     din_group,
                                     bias_group,
                                     dout_group,
                                     m,
                                     n,
                                     k,
                                     flag_bias,
                                     false,
                                     scale_group,
                                     act_param,
                                     ctx);
        } else {
#endif
          gemm_prepack_int8(weights_group,
                            din_group,
                            bias_group,
                            dout_group,
                            m,
                            n,
                            k,
                            flag_bias,
                            GemmMBias,
                            false,
                            false,
                            scale_group,
                            act_param,
                            ctx);
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
        }
#endif
      }
    }
  }
}

template void conv1x1s1_gemm_int8<int8_t>(const int8_t* i_data,
                                          int8_t* o_data,
                                          int num,
                                          int oc,
                                          int oh,
                                          int ow,
                                          int ic,
                                          int ih,
                                          int win,
                                          const int8_t* weights,
                                          const float* bias,
                                          const operators::ConvParam& param,
                                          ARMContext* ctx,
                                          const float* scale);

template void conv1x1s1_gemm_int8<float>(const int8_t* i_data,
                                         float* o_data,
                                         int num,
                                         int oc,
                                         int oh,
                                         int ow,
                                         int ic,
                                         int ih,
                                         int win,
                                         const int8_t* weights,
                                         const float* bias,
                                         const operators::ConvParam& param,
                                         ARMContext* ctx,
                                         const float* scale);

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm
 * implementation
 */
void conv_im2col_gemm(const float* i_data,
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
  int hblock = get_hblock(ctx, m);
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
      if (n == 1) {
        sgemv(weights_group,
              dB,
              dout_group,
              false,
              m,
              k,
              0.f,
              flag_bias,
              bias_group,
              act_param,
              ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float* bias_ptr = new float[n];
#else
        float bias_ptr[n];   // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        sgemv(dB,
              weights_group,
              dout_group,
              true,
              n,
              k,
              0.f,
              flag_bias,
              bias_ptr,
              act_param,
              ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
#endif
      } else {
        int ldb = n;
        sgemm_prepack(false,
                      m,
                      n,
                      k,
                      weights_group,
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
}

#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
// fuse im2col and packB, only support 3x3s1p1
static void padding_zero(const int8_t* data_in,
                         int iw,
                         int ih,
                         int8_t* data_out) {
  int ow = iw + 2;
  int oh = ih + 2;
  memset(data_out, 0, ow);
  memset(data_out + (oh - 1) * ow, 0, ow);
  auto data_out_ptr = data_out + ow;
  for (int row = 0; row < ih; row++) {
    data_out_ptr[0] = 0;
    memcpy(data_out_ptr + 1, data_in + row * iw, iw);
    data_out_ptr[ow - 1] = 0;
    data_out_ptr += ow;
  }
}

#define transfer4x12(line1, line2, line3, line4, data_out)            \
  line_1_2_q = vtrnq_s8(line1, line2);                                \
  line_3_4_q = vtrnq_s8(line3, line4);                                \
  out0_8 = vtrnq_s16(vreinterpretq_s16_s8(line_1_2_q.val[0]),         \
                     vreinterpretq_s16_s8(line_3_4_q.val[0]));        \
  out1_8 = vtrnq_s16(vreinterpretq_s16_s8(line_1_2_q.val[1]),         \
                     vreinterpretq_s16_s8(line_3_4_q.val[1]));        \
  line_1_2_int32_4 = vtrnq_s32(vreinterpretq_s32_s16(out0_8.val[0]),  \
                               vreinterpretq_s32_s16(out1_8.val[0])); \
  line_3_4_int32_4 = vtrnq_s32(vreinterpretq_s32_s16(out0_8.val[1]),  \
                               vreinterpretq_s32_s16(out1_8.val[1])); \
  vst1q_s32(reinterpret_cast<int*>(data_out),                         \
            vcombine_s32(vget_low_s32(line_1_2_int32_4.val[0]),       \
                         vget_low_s32(line_3_4_int32_4.val[0])));     \
  vst1q_s32(reinterpret_cast<int*>(data_out + 16),                    \
            vcombine_s32(vget_low_s32(line_1_2_int32_4.val[1]),       \
                         vget_low_s32(line_3_4_int32_4.val[1])));     \
  vst1q_s32(reinterpret_cast<int*>(data_out + 32),                    \
            vcombine_s32(vget_high_s32(line_1_2_int32_4.val[0]),      \
                         vget_high_s32(line_3_4_int32_4.val[0])));

#define LOAD_36x12_K3(vec_k0, vec_k1, vec_k2, data_input) \
  vec_k0 = vld1q_s8((data_input));                        \
  vec_k1 = vextq_s8(vec_k0, vec_zero, 1);                 \
  vec_k2 = vextq_s8(vec_k0, vec_zero, 2);

#define LOAD_36x12_K3_B8(vec_k0, vec_k1, vec_k2, data_input)                \
  vec_h = vld1_s8((data_input));                                            \
  vec_tmp = vld1_s8((data_input) + 8);                                      \
  vec_k0 = vcombine_s8(vec_h, vtbl1_s8(vec_tmp, vec_idx));                  \
  vec_k1 =                                                                  \
      vcombine_s8(vext_s8(vec_h, vec_tmp, 1), vtbl1_s8(vec_tmp, vec_idx1)); \
  vec_k2 = vcombine_s8(vext_s8(vec_h, vec_tmp, 2), vtbl1_s8(vec_tmp, vec_idx2));

#define LOAD_36x12_K3_B4(vec_k0, vec_k1, vec_k2, data_input)           \
  vec_h = vld1_s8((data_input));                                       \
  vec_tmp = vld1_s8((data_input) + 8);                                 \
  vec_k0 = vcombine_s8(vext_s8(vtbl1_s8(vec_h, vec_idx3), vec_tmp, 2), \
                       vext_s8(vec_tmp, vec_zero_h, 2));               \
  vec_k1 = vcombine_s8(vext_s8(vtbl1_s8(vec_h, vec_idx4), vec_tmp, 3), \
                       vext_s8(vec_tmp, vec_zero_h, 3));               \
  vec_k2 = vcombine_s8(vext_s8(vtbl1_s8(vec_h, vec_idx5), vec_tmp, 4), \
                       vext_s8(vec_tmp, vec_zero_h, 4));

#define transfer4x8(line1, line2, line3, line4, data_out)                \
  line_1_2 = vtrn_s8(line1, line2);                                      \
  line_3_4 = vtrn_s8(line3, line4);                                      \
  out0 = vtrn_s16(vreinterpret_s16_s8(line_1_2.val[0]),                  \
                  vreinterpret_s16_s8(line_3_4.val[0]));                 \
  out1 = vtrn_s16(vreinterpret_s16_s8(line_1_2.val[1]),                  \
                  vreinterpret_s16_s8(line_3_4.val[1]));                 \
  line_1_2_int32 = vtrn_s32(vreinterpret_s32_s16(out0.val[0]),           \
                            vreinterpret_s32_s16(out1.val[0]));          \
  line_3_4_int32 = vtrn_s32(vreinterpret_s32_s16(out0.val[1]),           \
                            vreinterpret_s32_s16(out1.val[1]));          \
  vst1q_s32(reinterpret_cast<int*>(data_out),                            \
            vcombine_s32(line_1_2_int32.val[0], line_3_4_int32.val[0])); \
  vst1q_s32(reinterpret_cast<int*>(data_out + 16),                       \
            vcombine_s32(line_1_2_int32.val[1], line_3_4_int32.val[1]));

#define LOAD_36x8_K3(vec_k0, vec_k1, vec_k2, data_input) \
  vec_k0 = vld1_s8((data_input));                        \
  vec_h = vld1_s8((data_input) + 8);                     \
  vec_k1 = vext_s8(vec_k0, vec_h, 1);                    \
  vec_k2 = vext_s8(vec_k0, vec_h, 2);

#define LOAD_36x8_K3_B4(vec_k0, vec_k1, vec_k2, data_input) \
  vec_h = vld1_s8((data_input));                            \
  vec_tmp = vld1_s8((data_input) + 8);                      \
  vec_k0 = vext_s8(vtbl1_s8(vec_h, vec_idx3), vec_tmp, 2);  \
  vec_k1 = vext_s8(vtbl1_s8(vec_h, vec_idx4), vec_tmp, 3);  \
  vec_k2 = vext_s8(vtbl1_s8(vec_h, vec_idx5), vec_tmp, 4);

#define transfer4x4(line1, line2, line3, line4, data_out) \
  cnt = 0;                                                \
  for (int i = 0; i < 4; i++) {                           \
    res[cnt++] = line1[i];                                \
    res[cnt++] = line2[i];                                \
    res[cnt++] = line3[i];                                \
    res[cnt++] = line4[i];                                \
  }                                                       \
  vst1q_s8(data_out, res);

#define LOAD_36x4                     \
  out_k00 = data_input;               \
  out_k01 = data_input + 1;           \
  out_k02 = data_input + 2;           \
  out_k10 = data_input + win;         \
  out_k11 = data_input + win + 1;     \
  out_k12 = data_input + win + 2;     \
  out_k20 = data_input + 2 * win;     \
  out_k21 = data_input + 2 * win + 1; \
  out_k22 = data_input + 2 * win + 2;

/* condition: ow % 8 == 0, ic % 4 == 0, 3x3s1p1d1g1
 workspace needs 4 * (width+2) * (height+2) + 16bytes(overread)*/
void im2col_packb_3x3s1p1(const int8_t* data_im,
                          int channels,
                          int height,
                          int width,
                          int8_t* data_col,
                          int8_t* workspace) {
  const int channel_size = height * width;
  const int kernel_size = 9;
  const int step = channels * kernel_size;
  const int loopw = channel_size;
  const int looph = ROUNDUP(step, 4);
  const int win = width + 2;
  const int hin = height + 2;
  const int padding_size = win * hin;
  const int output_w = width;
  int cnt = 0;
  int cur_ow, cur_oh, input_idx;
  int8_t* data_input;
  int8_t* data_output;
  int8_t *out_k00, *out_k01, *out_k02, *out_k10, *out_k11, *out_k12, *out_k20,
      *out_k21, *out_k22;
  int8x16_t vec_zero = vdupq_n_s8(0);
  int8x8_t vec_zero_h = vdup_n_s8(0);
  int8x8_t vec_idx = {2, 3, 4, 5, 2, 3, 4, 5};  // for vtbl1
  int8x8_t vec_idx1 = {3, 4, 5, 6, 3, 4, 5, 6};
  int8x8_t vec_idx2 = {4, 5, 6, 7, 4, 5, 6, 7};
  int8x8_t vec_idx3 = {0, 0, 0, 1, 2, 3, 6, 7};
  int8x8_t vec_idx4 = {0, 0, 0, 1, 2, 3, 4, 7};
  int8x8_t vec_idx5 = {0, 0, 0, 0, 2, 3, 4, 5};
  int8x16_t vec_k00, vec_k01, vec_k02, vec_k10, vec_k11, vec_k12, vec_k20,
      vec_k21, vec_k22;
  int8x16x2_t line_1_2_q, line_3_4_q;
  int16x8x2_t out0_8, out1_8;
  int32x4x2_t line_1_2_int32_4, line_3_4_int32_4;
  int8x8_t vec_k00_h, vec_k01_h, vec_k02_h, vec_k10_h, vec_k11_h, vec_k12_h,
      vec_k20_h, vec_k21_h, vec_k22_h;
  int8x8x2_t line_1_2, line_3_4;
  int16x4x2_t out0, out1;
  int32x2x2_t line_1_2_int32, line_3_4_int32;
  int8x8_t vec_h, vec_tmp, vec_tmp1;
  int8x16_t res;

  // 4c x 3 x 3, only support c%4==0
  for (int loop_row = 0; loop_row + 35 < looph; loop_row += 36) {
    int cur_ic = loop_row / kernel_size;
    // padding 4C per cycle
    for (int c = 0; c < 4; c++)
      padding_zero(data_im + (cur_ic + c) * channel_size,
                   width,
                   height,
                   workspace + c * padding_size);
    auto data_im_block = workspace;
    int loop_col = 0;  // only support ow % 8 == 0,
                       // tail must be 12/8/4
    for (; loop_col + 11 < loopw; loop_col += 12) {
      auto data_out_block = data_col + loop_col * looph + 12 * loop_row;
      cur_ow = loop_col % output_w;
      cur_oh = loop_col / output_w;
      input_idx = cur_oh * win + cur_ow;
      data_input = data_im_block;
      data_output = data_out_block;
      if (cur_ow + 11 < output_w) {  // overread 2bytes(16(neon) -
                                     // 12(block) - 2(pad))
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x12_K3(vec_k00, vec_k01, vec_k02, data_input);
        LOAD_36x12_K3(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k00, vec_k01, vec_k02, vec_k10, data_output);
        data_output += 48;
        LOAD_36x12_K3(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k11, vec_k12, vec_k20, vec_k21, data_output);
        data_output += 48;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x12_K3(vec_k00, vec_k01, vec_k02, data_input);
        LOAD_36x12_K3(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k22, vec_k00, vec_k01, vec_k02, data_output);
        data_output += 48;
        LOAD_36x12_K3(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k10, vec_k11, vec_k12, vec_k20, data_output);
        data_output += 48;
        // OC3
        data_input = data_im_block + input_idx + padding_size * 2;
        LOAD_36x12_K3(vec_k00, vec_k01, vec_k02, data_input);
        LOAD_36x12_K3(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k21, vec_k22, vec_k00, vec_k01, data_output);
        data_output += 48;
        LOAD_36x12_K3(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k02, vec_k10, vec_k11, vec_k12, data_output);
        data_output += 48;
        // OC4
        data_input = data_im_block + input_idx + padding_size * 3;
        LOAD_36x12_K3(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k20, vec_k21, vec_k22, vec_k00, data_output);
        data_output += 48;
        LOAD_36x12_K3(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k01, vec_k02, vec_k10, vec_k11, data_output);
        data_output += 48;
        LOAD_36x12_K3(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k12, vec_k20, vec_k21, vec_k22, data_output);
        data_output += 48;
      } else if (cur_ow + 7 < output_w) {  // current line fetch 8
                                           // and next line fetch 4
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x12_K3_B8(vec_k00, vec_k01, vec_k02, data_input);
        LOAD_36x12_K3_B8(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k00, vec_k01, vec_k02, vec_k10, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k11, vec_k12, vec_k20, vec_k21, data_output);
        data_output += 48;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x12_K3_B8(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k22, vec_k00, vec_k01, vec_k02, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k10, vec_k11, vec_k12, data_input + win);
        LOAD_36x12_K3_B8(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k10, vec_k11, vec_k12, vec_k20, data_output);
        data_output += 48;
        // OC3
        data_input = data_im_block + input_idx + 2 * padding_size;
        LOAD_36x12_K3_B8(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k21, vec_k22, vec_k00, vec_k01, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k02, vec_k10, vec_k11, vec_k12, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        // OC4
        data_input = data_im_block + input_idx + 3 * padding_size;
        LOAD_36x12_K3_B8(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k20, vec_k21, vec_k22, vec_k00, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k01, vec_k02, vec_k10, vec_k11, data_output);
        data_output += 48;
        LOAD_36x12_K3_B8(vec_k20, vec_k21, vec_k22, data_input + 2 * win);
        transfer4x12(vec_k12, vec_k20, vec_k21, vec_k22, data_output);
        data_output += 48;
      } else {  // 4, current line fetch 4 and next
                // line fetch 8
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x12_K3_B4(vec_k00, vec_k01, vec_k02, data_input);
        LOAD_36x12_K3_B4(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k00, vec_k01, vec_k02, vec_k10, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k20, vec_k21, vec_k22, data_input + win * 2);
        transfer4x12(vec_k11, vec_k12, vec_k20, vec_k21, data_output);
        data_output += 48;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x12_K3_B4(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k22, vec_k00, vec_k01, vec_k02, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k10, vec_k11, vec_k12, data_input + win);
        LOAD_36x12_K3_B4(vec_k20, vec_k21, vec_k22, data_input + win * 2);
        transfer4x12(vec_k10, vec_k11, vec_k12, vec_k20, data_output);
        data_output += 48;
        // OC3
        data_input = data_im_block + input_idx + 2 * padding_size;
        LOAD_36x12_K3_B4(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k21, vec_k22, vec_k00, vec_k01, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k02, vec_k10, vec_k11, vec_k12, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k20, vec_k21, vec_k22, data_input + win * 2);
        // OC4
        data_input = data_im_block + input_idx + 3 * padding_size;
        LOAD_36x12_K3_B4(vec_k00, vec_k01, vec_k02, data_input);
        transfer4x12(vec_k20, vec_k21, vec_k22, vec_k00, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k10, vec_k11, vec_k12, data_input + win);
        transfer4x12(vec_k01, vec_k02, vec_k10, vec_k11, data_output);
        data_output += 48;
        LOAD_36x12_K3_B4(vec_k20, vec_k21, vec_k22, data_input + win * 2);
        transfer4x12(vec_k12, vec_k20, vec_k21, vec_k22, data_output);
        data_output += 48;
      }
    }
    for (; loop_col + 7 < loopw; loop_col += 8) {
      auto data_out_block = data_col + loop_col * looph + 8 * loop_row;
      cur_ow = loop_col % output_w;
      cur_oh = loop_col / output_w;
      input_idx = cur_oh * win + cur_ow;
      data_output = data_out_block;
      data_input = data_im_block;
      if (cur_ow + 7 < output_w) {
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x8_K3(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        LOAD_36x8_K3(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        transfer4x8(vec_k00_h, vec_k01_h, vec_k02_h, vec_k10_h, data_output);
        data_output += 32;
        LOAD_36x8_K3(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k11_h, vec_k12_h, vec_k20_h, vec_k21_h, data_output);
        data_output += 32;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x8_K3(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k22_h, vec_k00_h, vec_k01_h, vec_k02_h, data_output);
        data_output += 32;
        LOAD_36x8_K3(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        LOAD_36x8_K3(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k10_h, vec_k11_h, vec_k12_h, vec_k20_h, data_output);
        data_output += 32;
        // OC3
        data_input = data_im_block + input_idx + 2 * padding_size;
        LOAD_36x8_K3(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k21_h, vec_k22_h, vec_k00_h, vec_k01_h, data_output);
        data_output += 32;
        LOAD_36x8_K3(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        LOAD_36x8_K3(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k02_h, vec_k10_h, vec_k11_h, vec_k12_h, data_output);
        data_output += 32;
        // OC4
        data_input = data_im_block + input_idx + 3 * padding_size;
        LOAD_36x8_K3(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k20_h, vec_k21_h, vec_k22_h, vec_k00_h, data_output);
        data_output += 32;
        LOAD_36x8_K3(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        LOAD_36x8_K3(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k01_h, vec_k02_h, vec_k10_h, vec_k11_h, data_output);
        data_output += 32;
        transfer4x8(vec_k12_h, vec_k20_h, vec_k21_h, vec_k22_h, data_output);
        data_output += 32;
      } else {
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x8_K3_B4(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        LOAD_36x8_K3_B4(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        transfer4x8(vec_k00_h, vec_k01_h, vec_k02_h, vec_k10_h, data_output);
        data_output += 32;
        LOAD_36x8_K3_B4(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k11_h, vec_k12_h, vec_k20_h, vec_k21_h, data_output);
        data_output += 32;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x8_K3_B4(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k22_h, vec_k00_h, vec_k01_h, vec_k02_h, data_output);
        data_output += 32;
        LOAD_36x8_K3_B4(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        LOAD_36x8_K3_B4(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k10_h, vec_k11_h, vec_k12_h, vec_k20_h, data_output);
        data_output += 32;
        // OC3
        data_input = data_im_block + input_idx + padding_size * 2;
        LOAD_36x8_K3_B4(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k21_h, vec_k22_h, vec_k00_h, vec_k01_h, data_output);
        data_output += 32;
        LOAD_36x8_K3_B4(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        LOAD_36x8_K3_B4(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k02_h, vec_k10_h, vec_k11_h, vec_k12_h, data_output);
        data_output += 32;
        // OC4
        data_input = data_im_block + input_idx + padding_size * 3;
        LOAD_36x8_K3_B4(vec_k00_h, vec_k01_h, vec_k02_h, data_input);
        transfer4x8(vec_k20_h, vec_k21_h, vec_k22_h, vec_k00_h, data_output);
        data_output += 32;
        LOAD_36x8_K3_B4(vec_k10_h, vec_k11_h, vec_k12_h, data_input + win);
        transfer4x8(vec_k01_h, vec_k02_h, vec_k10_h, vec_k11_h, data_output);
        data_output += 32;
        LOAD_36x8_K3_B4(vec_k20_h, vec_k21_h, vec_k22_h, data_input + 2 * win);
        transfer4x8(vec_k12_h, vec_k20_h, vec_k21_h, vec_k22_h, data_output);
        data_output += 32;
      }
    }
    for (; loop_col + 3 < loopw; loop_col += 4) {
      auto data_out_block = data_col + loop_col * looph + 4 * loop_row;
      cur_ow = loop_col % output_w;
      cur_oh = loop_col / output_w;
      input_idx = cur_oh * win + cur_ow;
      data_input = data_im_block;
      data_output = data_out_block;
      if (cur_ow + 3 < output_w) {
        // OC1
        data_input = data_im_block + input_idx;
        LOAD_36x4;
        transfer4x4(out_k00, out_k01, out_k02, out_k10, data_output);
        data_output += 16;
        transfer4x4(out_k11, out_k12, out_k20, out_k21, data_output);
        data_output += 16;
        // OC2
        data_input = data_im_block + input_idx + padding_size;
        LOAD_36x4;
        transfer4x4(out_k22, out_k00, out_k01, out_k02, data_output);
        data_output += 16;
        transfer4x4(out_k10, out_k11, out_k12, out_k20, data_output);
        data_output += 16;
        // OC3
        data_input = data_im_block + input_idx + 2 * padding_size;
        LOAD_36x4;
        transfer4x4(out_k21, out_k22, out_k00, out_k01, data_output);
        data_output += 16;
        transfer4x4(out_k02, out_k10, out_k11, out_k12, data_output);
        data_output += 16;
        // OC4
        data_input = data_im_block + input_idx + 3 * padding_size;
        LOAD_36x4;
        transfer4x4(out_k20, out_k21, out_k22, out_k00, data_output);
        data_output += 16;
        transfer4x4(out_k01, out_k02, out_k10, out_k11, data_output);
        data_output += 16;
        transfer4x4(out_k12, out_k20, out_k21, out_k22, data_output);
        data_output += 16;
      }
    }
  }
}

#undef transfer4x12
#undef transfer4x8
#undef transfer4x4
#undef LOAD_36x12_K3
#undef LOAD_36x12_K3_B8
#undef LOAD_36x12_K3_B4
#undef LOAD_36x8_K3
#undef LOAD_36x8_K3_B4
#undef LOAD_36x4

template <typename Dtype>
void conv_im2col_gemm_int8_fast(const int8_t* i_data,
                                Dtype* o_data,
                                int num,
                                int oc,
                                int oh,
                                int ow,
                                int ic,
                                int ih,
                                int win,
                                const int8_t* weights,
                                const float* bias,
                                const operators::ConvParam& param,
                                ARMContext* ctx,
                                const float* scale) {
  int group = 1;
  auto filter_dims = param.filter->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  int weights_size_per_group = m * k;
  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  if (n > 1 && m > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }

  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);
  auto pad_b = static_cast<int8_t*>(
      TargetMalloc(TARGET(kARM), 4 * (ih + 2) * (win + 2) + 16));

  for (int b = 0; b < num; ++b) {
    Dtype* dout_group = o_data + (b * oc) * channel_size_out;
    const int8_t* din_group =
        static_cast<const int8_t*>(i_data) + (b * ic) * channel_size_in;
    const int8_t* weights_group = static_cast<const int8_t*>(weights);
    const float* bias_group = bias;
    int8_t* dB = tmp_work_space;
    const float* scale_group = scale;

    im2col_packb_3x3s1p1(din_group, chin_per_group, ih, win, dB, pad_b);
    gemm_prepack_int8_nopack(weights_group,
                             dB,
                             bias_group,
                             dout_group,
                             m,
                             n,
                             k,
                             flag_bias,
                             false,
                             scale_group,
                             act_param,
                             ctx);
  }
  TargetFree(TARGET(kARM), pad_b);
}
#endif

template <typename Dtype>
void conv_im2col_gemm_int8(const int8_t* i_data,
                           Dtype* o_data,
                           int num,
                           int oc,
                           int oh,
                           int ow,
                           int ic,
                           int ih,
                           int win,
                           const int8_t* weights,
                           const float* bias,
                           const operators::ConvParam& param,
                           ARMContext* ctx,
                           const float* scale) {
  int group = param.groups;
  auto filter_dims = param.filter->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;
  int weights_size_per_group = m * k;
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
  if (ctx->has_sve2_i8mm()) {
    int hblock_sve = sve::get_hblock_int8_sve(ctx);
    int k_roundup_sve = ROUNDUP(k, sve::KBLOCK_INT8_SVE);
    int m_roundup_sve = ROUNDUP(m, hblock_sve);
    if (n > 1 && m > 1) {
      weights_size_per_group = ((m_roundup_sve * k_roundup_sve + 15) / 16) * 16;
    }
  } else {
#endif
    int hblock = get_hblock_int8(ctx);
    int k_roundup = ROUNDUP(k, KBLOCK_INT8);
    int m_roundup = ROUNDUP(m, hblock);
    if (n > 1 && m > 1) {
      weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
    }
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
  }
#endif

  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);

#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
  // only support 3x3s1p1d1 sdot, win == ow && ih ==
  // oh
  bool ker_3 = (kernel_h == kernel_w) && (kernel_w == 3);
  bool stride_1 = (stride_h == stride_w) && (stride_h == 1);
  bool dila_1 = (dila_h == dila_w) && (dila_h == 1);
  bool pad_1 = (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
               (paddings[0] == paddings[2]) && (paddings[0] == 1);
  bool mod_cond = (ic % 4 == 0) && (ow % 8 == 0);
  bool group_1 = (group == 1);
  bool mn_gt_1 = (m > 1) && (n > 1);
  if (ker_3 && stride_1 && dila_1 && pad_1 && group_1 && mn_gt_1 && mod_cond &&
      ctx->has_dot() && !ctx->has_sve2_i8mm()) {
    conv_im2col_gemm_int8_fast(i_data,
                               o_data,
                               num,
                               oc,
                               oh,
                               ow,
                               ic,
                               ih,
                               win,
                               weights,
                               bias,
                               param,
                               ctx,
                               scale);
    return;
  }
#endif
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      Dtype* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const int8_t* din_group = static_cast<const int8_t*>(i_data) +
                                (b * ic + g * chin_per_group) * channel_size_in;
      const int8_t* weights_group =
          static_cast<const int8_t*>(weights) + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      int8_t* dB = tmp_work_space;
      const float* scale_group = scale + g * m;
      im2col<int8_t>(din_group,
                     chin_per_group,
                     ih,
                     win,
                     kernel_h,
                     kernel_w,
                     pad_h,
                     paddings[1],
                     pad_w,
                     paddings[3],
                     stride_h,
                     stride_w,
                     dila_h,
                     dila_w,
                     dB);
      if (n == 1) {
        gemv_int8(weights_group,
                  dB,
                  dout_group,
                  false,
                  m,
                  k,
                  scale_group,
                  flag_bias,
                  bias_group,
                  act_param,
                  ctx);
      } else if (m == 1) {
#ifdef TARGET_IOS
        float* bias_ptr = new float[n];
        float* scale_ptr = new float[n];
#else
        float bias_ptr[n];   // NOLINT
        float scale_ptr[n];  // NOLINT
#endif
        if (flag_bias) {
          for (int i = 0; i < n; i++) {
            bias_ptr[i] = bias_group[0];
          }
        }
        for (int i = 0; i < n; i++) {
          scale_ptr[i] = scale_group[0];
        }
        gemv_int8(dB,
                  weights_group,
                  dout_group,
                  true,
                  n,
                  k,
                  scale_ptr,
                  flag_bias,
                  bias_ptr,
                  act_param,
                  ctx);
#ifdef TARGET_IOS
        delete[] bias_ptr;
        delete[] scale_ptr;
#endif
      } else {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
        if (ctx->has_sve2_i8mm()) {
          sve::gemm_prepack_int8_sve(weights_group,
                                     dB,
                                     bias_group,
                                     dout_group,
                                     m,
                                     n,
                                     k,
                                     flag_bias,
                                     false,
                                     scale_group,
                                     act_param,
                                     ctx);
        } else {
#endif
          gemm_prepack_int8(weights_group,
                            dB,
                            bias_group,
                            dout_group,
                            m,
                            n,
                            k,
                            flag_bias,
                            GemmMBias,
                            false,
                            false,
                            scale_group,
                            act_param,
                            ctx);
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
        }
#endif
      }
    }
  }
}

template void conv_im2col_gemm_int8<int8_t>(const int8_t* i_data,
                                            int8_t* o_data,
                                            int num,
                                            int oc,
                                            int oh,
                                            int ow,
                                            int ic,
                                            int ih,
                                            int win,
                                            const int8_t* weights,
                                            const float* bias,
                                            const operators::ConvParam& param,
                                            ARMContext* ctx,
                                            const float* scale);

template void conv_im2col_gemm_int8<float>(const int8_t* i_data,
                                           float* o_data,
                                           int num,
                                           int oc,
                                           int oh,
                                           int ow,
                                           int ic,
                                           int ih,
                                           int win,
                                           const int8_t* weights,
                                           const float* bias,
                                           const operators::ConvParam& param,
                                           ARMContext* ctx,
                                           const float* scale);

void conv_depthwise_3x3_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  int stride = param.strides[1];
  int pad = pad_w;
  bool flag_bias = param.bias != nullptr;
  bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
  if (stride == 1) {
    if (pads_less && (pad_h == pad_w) && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s1_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s1_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else if (stride == 2) {
    if (pads_less && pad_h == pad_w && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s2_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s2_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else {
    LOG(FATAL) << "fp32 depthwise conv3x3 stride: " << stride << " unsupported";
  }
}

void conv_depthwise_5x5_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  ctx->ExtendWorkspace((w_in + w_out + 16) * sizeof(float));
  if (stride == 2) {
    if (pad_h == pad_w && pad_h == 2 &&
        static_cast<int>(act_param.active_type) < 4 && w_in > 16) {
      // only support conv + relu/relu6
      conv_depthwise_5x5s2p2_fp32(reinterpret_cast<float*>(dout),
                                  reinterpret_cast<const float*>(din),
                                  reinterpret_cast<const float*>(weights),
                                  bias,
                                  flag_bias,
                                  num,
                                  ch_out,
                                  h_out,
                                  w_out,
                                  ch_in,
                                  h_in,
                                  w_in,
                                  param,
                                  ctx);
    } else {
      conv_depthwise_5x5s2_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else if (stride == 1) {
    if (0 && pad_h == pad_w && pad_h == 2 &&
        static_cast<int>(act_param.active_type) < 4 && w_in > 8) {
      // only support conv + relu/relu6
      conv_depthwise_5x5s1p2_fp32(reinterpret_cast<float*>(dout),
                                  reinterpret_cast<const float*>(din),
                                  reinterpret_cast<const float*>(weights),
                                  bias,
                                  flag_bias,
                                  flag_relu,
                                  num,
                                  ch_in,
                                  h_in,
                                  w_in,
                                  h_out,
                                  w_out,
                                  param,
                                  ctx);
    } else {
      conv_depthwise_5x5s1_fp32(reinterpret_cast<float*>(dout),
                                reinterpret_cast<const float*>(din),
                                reinterpret_cast<const float*>(weights),
                                bias,
                                flag_bias,
                                flag_relu,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                param,
                                ctx);
    }
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv";
  }
}

void conv_depthwise_3x3_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  bool support_act_type = flag_act <= 2;
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (param.strides[0] == 1 && param.strides[1] == 1);
  bool support_width_type = w_in > 9 ? true : false;
  if (stride == 1) {
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      conv_depthwise_3x3s1_int8(reinterpret_cast<float*>(dout),
                                reinterpret_cast<const int8_t*>(din),
                                reinterpret_cast<const int8_t*>(weights),
                                scale,
                                bias,
                                flag_bias,
                                flag_act,
                                alpha,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                ctx);
    } else {
      conv_depthwise_3x3s1_int8_float_impl(
          reinterpret_cast<float*>(dout),
          reinterpret_cast<const int8_t*>(din),
          reinterpret_cast<const int8_t*>(weights),
          scale,
          bias,
          flag_bias,
          flag_act,
          alpha,
          num,
          ch_in,
          h_in,
          w_in,
          h_out,
          w_out,
          pad_w,
          pad_h,
          ctx);
    }
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv int8";
  }
}

void conv_depthwise_3x3_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  bool support_act_type = flag_act <= 2;
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (param.strides[0] == 1 && param.strides[1] == 1);
  bool support_width_type = w_in > 9 ? true : false;
  if (stride == 1) {
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      conv_depthwise_3x3s1_int8(reinterpret_cast<int8_t*>(dout),
                                reinterpret_cast<const int8_t*>(din),
                                reinterpret_cast<const int8_t*>(weights),
                                scale,
                                bias,
                                flag_bias,
                                flag_act,
                                alpha,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                ctx);
    } else {
      conv_depthwise_3x3s1_int8_int8_impl(
          reinterpret_cast<int8_t*>(dout),
          reinterpret_cast<const int8_t*>(din),
          reinterpret_cast<const int8_t*>(weights),
          scale,
          bias,
          flag_bias,
          flag_act,
          alpha,
          num,
          ch_in,
          h_in,
          w_in,
          h_out,
          w_out,
          pad_w,
          pad_h,
          ctx);
    }
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv int8";
  }
}

void conv_depthwise_5x5_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_5x5s2_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv int8";
  }
}

void conv_depthwise_5x5_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_5x5s2_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
                              bias,
                              flag_bias,
                              flag_act,
                              alpha,
                              num,
                              ch_in,
                              h_in,
                              w_in,
                              h_out,
                              w_out,
                              pad_w,
                              pad_h,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv int8";
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
