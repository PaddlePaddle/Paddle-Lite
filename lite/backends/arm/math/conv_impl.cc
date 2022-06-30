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
