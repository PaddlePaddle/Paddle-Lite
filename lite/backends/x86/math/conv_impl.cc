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

#include "lite/backends/x86/math/conv_impl.h"
#include <immintrin.h>
#include <algorithm>
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace x86 {
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
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
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
#ifdef __AVX__
          for (; ow + 7 < ow_end; ow += 8, iw += 8) {
            __m256 vtmp = _mm256_load_ps(data_im_ptr + iw);
            _mm256_storeu_ps(data_col_ptr + ow, vtmp);
          }
#else
          for (; ow + 3 < ow_end; ow += 4, iw += 4) {
            __m128 vtmp = _mm_load_ps(data_im_ptr + iw);
            _mm_storeu_ps(data_col_ptr + ow, vtmp);
          }
#endif
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
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
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
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
            __m128 vtmp0 = _mm_load_ps(data_im_ptr + iw);
            __m128 vtmp1 = _mm_load_ps(data_im_ptr + iw + 4);
            __m128 vres;
            vres[0] = vtmp0[0];
            vres[1] = vtmp0[2];
            vres[2] = vtmp1[0];
            vres[3] = vtmp1[2];
            _mm_storeu_ps(data_col_ptr + ow, vtmp);
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
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
                    X86Context* ctx) {
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  auto act_param = param.activation_param;

  int hblock = get_hblock(ctx);
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
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else if (m == 1) {
        float bias_ptr[n];  // NOLINT
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
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
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
                      X86Context* ctx) {
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
  int hblock = get_hblock(ctx);
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
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
      } else if (m == 1) {
        float bias_ptr[n];  // NOLINT
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
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
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

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
