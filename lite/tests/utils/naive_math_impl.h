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

#pragma once
#include <cmath>

template <typename type>
static void basic_trans_mat_to_c4(const type* input,
                                  type* output,
                                  const int ldin,
                                  const int M,
                                  const int K,
                                  bool pack_k) {
  const int m_round = (M + 3) / 4 * 4;
  int k_round = (K + 3) / 4 * 4;
  if (!pack_k) {
    k_round = K;
  }
  const int m_loop = m_round / 4;
  type* zero_buf = new type[K];
  memset(zero_buf, 0, K * sizeof(type));
  for (int i = 0; i < m_loop; ++i) {
    const type* in0 = input + i * 4 * ldin;
    const type* in1 = in0 + ldin;
    const type* in2 = in1 + ldin;
    const type* in3 = in2 + ldin;
    if (4 * (i + 1) - M > 0) {
      switch (4 * (i + 1) - M) {
        case 3:
          in1 = zero_buf;
        case 2:
          in2 = zero_buf;
        case 1:
          in3 = zero_buf;
        default:
          break;
      }
    }
    for (int j = 0; j < K; ++j) {
      *output++ = *in0++;
      *output++ = *in1++;
      *output++ = *in2++;
      *output++ = *in3++;
    }
    for (int j = K; j < k_round; ++j) {
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
    }
  }
  delete[] zero_buf;
}
template <typename type>
static void basic_trans_mat_to_c8(const type* input,
                                  type* output,
                                  const int ldin,
                                  const int M,
                                  const int K,
                                  bool pack_k) {
  const int m_round = (M + 7) / 8 * 8;
  int k_round = (K + 7) / 8 * 8;
  if (!pack_k) {
    k_round = K;
  }
  const int m_loop = m_round / 8;
  type zero_buf[K];
  memset(zero_buf, 0, K * sizeof(type));
  for (int i = 0; i < m_loop; ++i) {
    const type* in0 = input + i * 8 * ldin;
    const type* in1 = in0 + ldin;
    const type* in2 = in1 + ldin;
    const type* in3 = in2 + ldin;
    const type* in4 = in3 + ldin;
    const type* in5 = in4 + ldin;
    const type* in6 = in5 + ldin;
    const type* in7 = in6 + ldin;
    if (8 * (i + 1) - M > 0) {
      switch (8 * (i + 1) - M) {
        case 7:
          in1 = zero_buf;
        case 6:
          in2 = zero_buf;
        case 5:
          in3 = zero_buf;
        case 4:
          in4 = zero_buf;
        case 3:
          in5 = zero_buf;
        case 2:
          in6 = zero_buf;
        case 1:
          in7 = zero_buf;
        default:
          break;
      }
    }
    for (int j = 0; j < K; ++j) {
      *output++ = *in0++;
      *output++ = *in1++;
      *output++ = *in2++;
      *output++ = *in3++;
      *output++ = *in4++;
      *output++ = *in5++;
      *output++ = *in6++;
      *output++ = *in7++;
    }
    for (int j = K; j < k_round; ++j) {
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
      *output++ = static_cast<type>(0);
    }
  }
}

template <typename type, typename type2>
static void basic_gemm_c4(bool trans_a,
                          bool trans_b,
                          int m,
                          int n,
                          int k,
                          type2 alpha,
                          const type* a,
                          int lda,
                          const type* b,
                          int ldb,
                          type2 beta,
                          type2* c,
                          int ldc,
                          const type2* bias,
                          bool flag_bias = false,
                          bool flag_relu = false) {
  type2* tmp_c = reinterpret_cast<type2*>(malloc(m * ldc * sizeof(type2)));
  memset(tmp_c, 0, m * ldc * sizeof(type2));
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < m; ++i) {
    auto bias_data = static_cast<type2>(0);
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      auto sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * lda + i];
        } else {
          av = a[i * lda + l];
        }
        if (trans_b) {
          bv = b[j * ldb + l];
        } else {
          bv = b[l * ldb + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * tmp_c[i * ldc + j] + bias_data;
      if (flag_relu) {
        tmp_c[i * ldc + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        tmp_c[i * ldc + j] = tmp;
      }
    }
  }
  //! trans c to c4
  basic_trans_mat_to_c4(tmp_c, c, ldc, m, n, false);
  free(tmp_c);
}

template <typename type, typename type2>
static void basic_gemm_c8(bool trans_a,
                          bool trans_b,
                          int m,
                          int n,
                          int k,
                          type2 alpha,
                          const type* a,
                          int lda,
                          const type* b,
                          int ldb,
                          type2 beta,
                          type2* c,
                          int ldc,
                          const type2* bias,
                          bool flag_bias = false,
                          bool flag_relu = false) {
  type2* tmp_c = reinterpret_cast<type2*>(malloc(m * ldc * sizeof(type2)));
  memset(tmp_c, 0, m * ldc * sizeof(type2));
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < m; ++i) {
    auto bias_data = static_cast<type2>(0);
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      auto sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * lda + i];
        } else {
          av = a[i * lda + l];
        }
        if (trans_b) {
          bv = b[j * ldb + l];
        } else {
          bv = b[l * ldb + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * tmp_c[i * ldc + j] + bias_data;
      if (flag_relu) {
        tmp_c[i * ldc + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        tmp_c[i * ldc + j] = tmp;
      }
    }
  }
  //! trans c to c4
  basic_trans_mat_to_c8(tmp_c, c, ldc, m, n, false);
  free(tmp_c);
}
template <typename type, typename type2>
static void basic_gemm(bool trans_a,
                       bool trans_b,
                       int m,
                       int n,
                       int k,
                       type2 alpha,
                       const type* a,
                       int lda,
                       const type* b,
                       int ldb,
                       type2 beta,
                       type2* c,
                       int ldc,
                       const type2* bias,
                       bool flag_bias = false,
                       bool flag_relu = false) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < m; ++i) {
    auto bias_data = static_cast<type2>(0);
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      auto sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * lda + i];
        } else {
          av = a[i * lda + l];
        }
        if (trans_b) {
          bv = b[j * ldb + l];
        } else {
          bv = b[l * ldb + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * c[i * ldc + j] + bias_data;
      if (flag_relu) {
        c[i * ldc + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        c[i * ldc + j] = tmp;
      }
    }
  }
}

template <typename type, typename type2>
static void basic_gemv(int m,
                       int k,
                       const type* a,
                       const type* b,
                       const type2* bias,
                       type2* c,
                       type2 alpha,
                       type2 beta,
                       bool trans_a = false,
                       bool flag_bias = false,
                       int flag_act = false,
                       float six = 6.f,
                       float leakey_relu_alpha = 1.f) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < m; ++i) {
    auto bias_data = static_cast<type2>(0);
    if (flag_bias) {
      bias_data = bias[i];
    }
    auto sum = static_cast<type2>(0);
    for (int j = 0; j < k; ++j) {
      type av;
      if (trans_a) {
        av = a[j * m + i];
      } else {
        av = a[i * k + j];
      }
      sum += av * b[j];
    }
    type2 tmp = alpha * sum + beta * c[i] + bias_data;
    if (flag_act > 0) {
      if (flag_act == 1) {  // relu
        c[i] = tmp > (type2)0 ? tmp : (type2)0;
      } else if (flag_act == 2) {  // relu 6
        c[i] = tmp > (type2)0 ? tmp : (type2)0;
        c[i] = c[i] < six ? c[i] : six;  // ut compute
      } else if (flag_act == 4) {        // leakey relu
        c[i] = tmp < (type2)0 ? (type2)(tmp * leakey_relu_alpha) : tmp;
      }
    } else {
      c[i] = tmp;
    }
  }
}

/**
 * \brief basic direct convolution function
 */
//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
static void conv_basic(const Dtype1* din,
                       Dtype2* dout,
                       int num,
                       int chout,
                       int hout,
                       int wout,
                       int chin,
                       int hin,
                       int win,
                       const Dtype1* weights,
                       const Dtype2* bias,
                       int group,
                       int kernel_w,
                       int kernel_h,
                       int stride_w,
                       int stride_h,
                       int dila_w,
                       int dila_h,
                       int pad_w,
                       int pad_h,
                       bool flag_bias,
                       int act_type,
                       float six = 6.f,
                       float scale = 1.f) {
  Dtype2 beta = 0;
  auto src_data = din;
  auto dst_data_ref = dout;
  auto weights_data = weights;
  auto with_bias = flag_bias;
  auto bias_data = bias;

  int in_num = num;
  int out_channels = chout;
  int out_h = hout;
  int out_w = wout;

  int in_channel = chin;
  int in_h = hin;
  int in_w = win;
  int out_c_group = out_channels / group;
  int in_c_group = in_channel / group;

  for (int n = 0; n < in_num; ++n) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(4)
#endif
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * group * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            Dtype2 bias_d = with_bias ? (bias_data[g * out_c_group + oc]) : 0;
            dst_data_ref[out_idx] = bias_d;  // + dst_data_ref[out_idx] * beta;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - pad_w + kw * (dila_w);
                  int ih = oh * stride_h - pad_h + kh * (dila_h);
                  if (iw < 0 || iw >= in_w) continue;
                  if (ih < 0 || ih >= in_h) continue;

                  int iidx = n * in_channel * in_h * in_w +
                             g * in_c_group * in_h * in_w + ic * in_h * in_w +
                             ih * in_w + iw;
                  int widx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;

                  dst_data_ref[out_idx] += src_data[iidx] * weights_data[widx];
                }
              }
            }
            if (act_type > 0) {
              // 1-relu 2-relu6 4-leakyrelu
              if (act_type == 1) {
                dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)0;
              } else if (act_type == 2) {
                dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)0;
                dst_data_ref[out_idx] = dst_data_ref[out_idx] < (Dtype2)six
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)six;
              } else if (act_type == 4) {
                dst_data_ref[out_idx] =
                    dst_data_ref[out_idx] > (Dtype2)0
                        ? dst_data_ref[out_idx]
                        : (Dtype2)(dst_data_ref[out_idx] * scale);
              } else {
                printf("this act type: %d does not support \n", act_type);
              }
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
static void fill_bias_relu(Dtype* tensor,
                           const Dtype* bias,
                           int channel,
                           int channel_size,
                           bool flag_bias,
                           bool flag_relu) {
  Dtype* data = tensor;
  for (int j = 0; j < channel; ++j) {
    Dtype bias_c = flag_bias ? bias[j] : 0;
    for (int i = 0; i < channel_size; i++) {
      data[i] += bias_c;
      if (flag_relu) {
        data[i] = data[i] > 0 ? data[i] : 0.f;
      }
    }
    data += channel_size;
  }
}

template <typename Dtype>
static void do_relu(Dtype* tensor, int size) {
  for (int j = 0; j < size; ++j) {
    tensor[j] = tensor[j] > 0 ? tensor[j] : (Dtype)0;
  }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
static void col2im(const Dtype* data_col,
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
                   Dtype* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(Dtype));
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

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
void deconv_basic(const Dtype1* din,
                  Dtype2* dout,
                  int num,
                  int chout,
                  int hout,
                  int wout,
                  int chin,
                  int hin,
                  int win,
                  const Dtype1* weights,
                  const Dtype2* bias,
                  int group,
                  int kernel_w,
                  int kernel_h,
                  int stride_w,
                  int stride_h,
                  int dila_w,
                  int dila_h,
                  int pad_w0,
                  int pad_w1,
                  int pad_h0,
                  int pad_h1,
                  bool flag_bias,
                  bool flag_relu) {
  int m = chout * kernel_w * kernel_h / group;
  int n = hin * win;
  int k = chin / group;

  int group_size_in = win * hin * chin / group;
  int group_size_coldata = m * n;
  int group_size_weights = chin * chout * kernel_w * kernel_h / (group * group);
  bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) &&
                      (stride_w == 1) && (pad_w0 == 0) && (pad_h0 == 0) &&
                      (pad_w1 == 0) && (pad_h1 == 0) && (dila_w == 1) &&
                      (dila_h == 1);

  Dtype2* workspace_ptr =
      static_cast<Dtype2*>(malloc(sizeof(float) * m * n * group));

  for (int i = 0; i < num; ++i) {
    const Dtype1* din_batch = din + i * chin * hin * win;
    Dtype2* dout_batch = dout + i * chout * hout * wout;

    Dtype2* col_data = workspace_ptr;
    if (flag_1x1s1p1) {
      col_data = dout_batch;
    }
    memset(col_data, 0, sizeof(Dtype2) * group_size_coldata * group);
    for (int g = 0; g < group; ++g) {
      const Dtype1* din_group = din_batch + g * group_size_in;
      const Dtype1* weights_group = weights + g * group_size_weights;
      Dtype2* coldata_group = col_data + g * group_size_coldata;
      basic_gemm<Dtype1, Dtype2>(true,
                                 false,
                                 m,
                                 n,
                                 k,
                                 1,
                                 weights_group,
                                 m,
                                 din_group,
                                 n,
                                 0,
                                 coldata_group,
                                 n,
                                 nullptr,
                                 false,
                                 (!flag_bias && flag_relu));
    }

    if (!flag_1x1s1p1) {
      col2im(col_data,
             chout,
             hout,
             wout,
             kernel_h,
             kernel_w,
             pad_h0,
             pad_h1,
             pad_w0,
             pad_w1,
             stride_h,
             stride_w,
             dila_h,
             dila_w,
             dout_batch);
    }
    //! add bias
    if (flag_bias) {
      fill_bias_relu(
          dout_batch, bias, chout, wout * hout, flag_bias, flag_relu);
    }
  }
  free(workspace_ptr);
}

float deformable_bilinear(const float* bottom_data,
                          const int data_width,
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

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
void deformable_conv_basic(const Dtype1* in_data,
                           const float* offset_data,
                           const float* mask_data,
                           Dtype2* out_data,
                           int num,
                           int chout,
                           int hout,
                           int wout,
                           int chin,
                           int hin,
                           int win,
                           const Dtype1* weights,
                           const Dtype2* bias,
                           int group,
                           int kernel_w,
                           int kernel_h,
                           int stride_w,
                           int stride_h,
                           int dila_w,
                           int dila_h,
                           int pad_w,
                           int pad_h,
                           bool flag_bias,
                           bool flag_relu,
                           bool modulated) {
  int out_c_group = chout / group;
  int in_c_group = chin / group;
  int in_size = hin * win;
  int out_size = hout * wout;
  int c_in_size = chin * in_size;
  int c_out_size = chout * out_size;
  int kernel_size = kernel_w * kernel_h;
  for (int n = 0; n < num; n++) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(4)
#endif
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < hout; oh++) {
          for (int ow = 0; ow < wout; ow++) {
            int out_idx = n * c_out_size + g * out_c_group * out_size +
                          oc * out_size + oh * wout + ow;
            Dtype2 bias_d = flag_bias ? bias[g * out_c_group + oc] : 0;
            out_data[out_idx] = bias_d + out_data[out_idx];
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int fh = 0; fh < kernel_h; fh++) {
                for (int fw = 0; fw < kernel_w; fw++) {
                  const float* offset_data_ptr =
                      offset_data + n * group * 2 * kernel_size * out_size +
                      g * 2 * kernel_size * out_size;
                  const int data_offset_h_ptr =
                      ((2 * (fh * kernel_w + fw)) * hout + oh) * wout + ow;
                  const int data_offset_w_ptr =
                      ((2 * (fh * kernel_w + fw) + 1) * hout + oh) * wout + ow;
                  const float offset_h = offset_data_ptr[data_offset_h_ptr];
                  const float offset_w = offset_data_ptr[data_offset_w_ptr];
                  const float iw =
                      ow * stride_w - pad_w + kernel_w * dila_w + offset_w;
                  const float ih =
                      oh * stride_h - pad_h + kernel_h * dila_h + offset_h;
                  if (ih >= 0 && ih < hin && iw >= 0 && iw < win) {
                    const float map_h = kernel_h * dila_h + offset_h;
                    const float map_w = kernel_w * dila_w + offset_w;
                    const int cur_height = hin - (oh * stride_h - pad_h);
                    const int cur_width = win - (ow * stride_w - pad_w);
                    const float* in_data_offset =
                        in_data + n * c_in_size +
                        (g * in_c_group + ic) * in_size +
                        (oh * stride_h - pad_h) * win + (ow * stride_w - pad_w);
                    float val = deformable_bilinear(in_data_offset,
                                                    win,
                                                    cur_height,
                                                    cur_width,
                                                    map_h,
                                                    map_w);

                    if (modulated) {
                      // use mask
                      const float* mask_ptr =
                          mask_data + n * group * kernel_size * out_size +
                          g * kernel_size * out_size +
                          (fh * kernel_w + fw) * hout * wout + oh * wout + ow;
                      val *= mask_ptr[0];
                    }
                    int widx = g * out_c_group * in_c_group * kernel_size +
                               oc * in_c_group * kernel_size +
                               ic * kernel_size + fh * kernel_w + fw;
                    out_data[out_idx] += val * weights[widx];
                  }
                }
              }
            }
            if (flag_relu) {
              out_data[out_idx] = out_data[out_idx] > 0 ? out_data[out_idx] : 0;
            }
          }
        }
      }
    }
  }
}
