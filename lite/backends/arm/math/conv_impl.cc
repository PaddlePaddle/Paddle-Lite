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
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/backends/arm/math/sgemv.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

/**
 * \brief neon implementation to add bias
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
void fill_bias(float* tensor,
               const float* bias,
               int channel,
               int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  float* data = tensor;

  for (int j = 0; j < channel; ++j) {
    float32x4_t vdata = vdupq_n_f32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_f32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

void fill_bias_int8(int* tensor,
                    const int* bias,
                    int channel,
                    int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  int* data = tensor;
  for (int j = 0; j < channel; ++j) {
    int32x4_t vdata = vdupq_n_s32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_s32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

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
void im2col(const Dtype* data_im,
            int channels,
            int height,
            int width,
            int kernel_h,
            int kernel_w,
            int pad_h,
            int pad_w,
            int stride_h,
            int stride_w,
            int dilation_h,
            int dilation_w,
            Dtype* data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
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

  int hblock = get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
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
              flag_bias,
              bias_group,
              flag_relu);
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
                      flag_relu,
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
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
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
                  flag_relu,
                  ctx);
      } else {
        gemm_prepack_int8(weights_group,
                          din_group,
                          bias_group,
                          dout_group,
                          m,
                          n,
                          k,
                          flag_bias,
                          flag_relu,
                          false,
                          scale_group,
                          ctx);
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
  int hblock = get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

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

      im2col(din_group,
             chin_per_group,
             ih,
             win,
             kernel_h,
             kernel_w,
             param.paddings[0],
             param.paddings[1],
             param.strides[0],
             param.strides[1],
             param.dilations[0],
             param.dilations[1],
             dB);

      if (n == 1) {
        sgemv(weights_group,
              dB,
              dout_group,
              false,
              m,
              k,
              flag_bias,
              bias_group,
              flag_relu);
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
                      flag_relu,
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
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = param.dilations[0];
  int dila_w = param.dilations[1];
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  int hblock = get_hblock_int8(ctx);
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }

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

      im2col(din_group,
             chin_per_group,
             ih,
             win,
             kernel_h,
             kernel_w,
             pad_h,
             pad_w,
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
                  flag_relu,
                  ctx);
      } else {
        gemm_prepack_int8(weights_group,
                          dB,
                          bias_group,
                          dout_group,
                          m,
                          n,
                          k,
                          flag_bias,
                          flag_relu,
                          false,
                          scale_group,
                          ctx);
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
  int stride = param.strides[1];
  if (stride == 1) {
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
                              ctx);
  } else if (stride == 2) {
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
                              ctx);
  } else {
    LOG(FATAL) << "fp32 depthwise conv3x3 stride: " << stride << " unsupported";
  }
#if 0
  if (pad == 1) {
    conv_depthwise_3x3p1_fp32(reinterpret_cast<const float*>(din),
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
                              stride,
                              flag_bias,
                              flag_relu,
                              ctx);
  } else if (pad == 0 && h_in > 2) {
    conv_depthwise_3x3p0_fp32(reinterpret_cast<const float*>(din),
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
                              stride,
                              flag_bias,
                              flag_relu,
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv";
  }
#endif
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
  int pad = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  ctx->ExtendWorkspace((w_in + w_out) * sizeof(float));
  if (pad == 2 && stride == 2) {
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
                              pad,
                              flag_bias,
                              flag_relu,
                              ctx);
  } else if (stride == 1) {
    conv_depthwise_5x5s1_fp32(reinterpret_cast<const float*>(din),
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
                              flag_relu,
                              ctx);
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
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  if (stride == 1) {
    conv_depthwise_3x3s1_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  if (stride == 1) {
    conv_depthwise_3x3s1_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
                              ctx);
  } else if (stride == 2) {
    conv_depthwise_3x3s2_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<float*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  if (stride == 1) {
    conv_depthwise_5x5s1_int8(reinterpret_cast<int8_t*>(dout),
                              reinterpret_cast<const int8_t*>(din),
                              reinterpret_cast<const int8_t*>(weights),
                              scale,
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
                              ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv int8";
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
