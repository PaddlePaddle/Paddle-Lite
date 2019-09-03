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
#pragma omp parallel for
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
                       bool flag_relu = false) {
#pragma omp parallel for
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
    if (flag_relu) {
      c[i] = tmp > (type2)0 ? tmp : (type2)0;
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
                       bool flag_relu) {
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
#pragma omp parallel for collapse(4)
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
            if (flag_relu) {
              dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                          ? dst_data_ref[out_idx]
                                          : (Dtype2)0;
            }
          }
        }
      }
    }
  }
}
