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
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/print_info.h"
#include "lite/tests/utils/tensor_utils.h"
#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/pool_compute.h"
#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif
#endif  // LITE_WITH_ARM

DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_bool(basic_test, false, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h, 1, "pad height");
DEFINE_int32(pad_w, 1, "pad width");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");

DEFINE_bool(ceil_mode, true, "do ceil_mode");
DEFINE_bool(flag_global, true, "global pooling");
DEFINE_bool(exclusive, true, "do exclusive");
DEFINE_bool(adaptive, false, "no do adaptive");
DEFINE_bool(use_quantizer, false, "no do use_quantizer");

DEFINE_string(pooling_type, "max", "do max pooling");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::PoolParam PoolParam;
using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::PoolParam& param) {
  DDim dim_out = dim_in;
  auto kernel_h = param.ksize[0];
  auto kernel_w = param.ksize[1];
  auto h = dim_in[2];
  auto w = dim_in[3];
  auto paddings = *param.paddings;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  bool ceil_mode = param.ceil_mode;
  bool flag_global = param.global_pooling;
  int hout = 1;
  int wout = 1;
  if (!flag_global) {
    if (!ceil_mode) {
      hout = (h - kernel_h + paddings[0] + paddings[1]) / stride_h + 1;
      wout = (w - kernel_w + paddings[2] + paddings[3]) / stride_w + 1;
    } else {
      hout =
          (h - kernel_h + paddings[0] + paddings[1] + stride_h - 1) / stride_h +
          1;
      wout =
          (w - kernel_w + paddings[2] + paddings[3] + stride_w - 1) / stride_w +
          1;
    }
  }
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
void pooling_basic(const Dtype1* din,
                   Dtype2* dout,
                   int num,
                   int chout,
                   int hout,
                   int wout,
                   int chin,
                   int hin,
                   int win,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool global_pooling,
                   bool exclusive,
                   bool adaptive,
                   bool ceil_mode,
                   bool use_quantizer,
                   const std::string& pooling_type) {
  // no need to pad input tensor, border is zero pad inside this function
  memset(dout, 0, num * chout * hout * wout * sizeof(Dtype2));
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;

  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        Dtype2* dout_batch = dout + n * chout * size_channel_out;
        const Dtype1* din_batch = din + n * chin * size_channel_in;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int c = 0; c < chout; ++c) {
          const Dtype1* din_ch = din_batch + c * size_channel_in;  // in address
          Dtype1 tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            Dtype1 tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        Dtype2* dout_batch = dout + n * chout * size_channel_out;
        const Dtype1* din_batch = din + n * chin * size_channel_in;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int c = 0; c < chout; ++c) {
          const Dtype1* din_ch = din_batch + c * size_channel_in;  // in address
          Dtype1 sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    for (int ind_n = 0; ind_n < num; ++ind_n) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int ind_c = 0; ind_c < chin; ++ind_c) {
        for (int ind_h = 0; ind_h < hout; ++ind_h) {
          int sh, eh;
          if (adaptive) {
            sh = AdaptStartIndex(ind_h, hin, hout);
            eh = AdaptEndIndex(ind_h, hin, hout);
          } else {
            sh = ind_h * stride_h;
            eh = sh + kernel_h;
            sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
            eh = (eh - pad_h) > hin ? hin : eh - pad_h;
          }
          for (int ind_w = 0; ind_w < wout; ++ind_w) {
            int sw, ew;
            if (adaptive) {
              sw = AdaptStartIndex(ind_w, win, wout);
              ew = AdaptEndIndex(ind_w, win, wout);
            } else {
              sw = ind_w * stride_w;
              ew = sw + kernel_w;
              sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
              ew = (ew - pad_w) > win ? win : ew - pad_w;
            }
            Dtype1 result = static_cast<Dtype1>(0);
            int dst_ind = (ind_n * chout + ind_c) * size_channel_out +
                          ind_h * wout + ind_w;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_ind =
                    (ind_n * chin + ind_c) * size_channel_in + kh * win + kw;
                if (kh == sh && kw == sw) {
                  result = din[src_ind];
                } else {
                  if (pooling_type == "max") {
                    result = result >= din[src_ind] ? result : din[src_ind];
                  } else if (pooling_type == "avg") {
                    result += din[src_ind];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                int div = (ew - sw) * (eh - sh);
                div = div > 0 ? div : 1;
                result /= div;
              } else {
                int bh = kernel_h;
                int bw = kernel_w;
                if (ew == win) {
                  bw = (sw + kernel_w) >= (win + paddings[3])
                           ? (win + paddings[3])
                           : (sw + kernel_w);
                  bw -= sw;
                  if ((sw - pad_w) < 0 &&
                      (sw + kernel_w) > (win + paddings[3])) {
                    bw += pad_w;
                  }
                }
                if (eh == hin) {
                  bh = (sh + kernel_h) >= (hin + paddings[1])
                           ? (hin + paddings[1])
                           : (sh + kernel_h);
                  bh -= sh;
                  if ((sh - pad_h) < 0 &&
                      (sh + kernel_h) > (hin + paddings[1])) {
                    bh += pad_h;
                  }
                }
                result /= bh * bw;
              }
            }
            dout[dst_ind] = result;
          }
        }
      }
    }
  }
}

void print_pool_success_or_fail_info(std::string op_name,
                                     bool has_success,
                                     const DDim dim_in,
                                     const DDim dim_out,
                                     std::vector<int> ksize,
                                     std::vector<int> pads,
                                     std::vector<int> strides,
                                     bool flag_global,
                                     std::string pooling_type,
                                     bool ceil_mode,
                                     bool exclusive,
                                     int th,
                                     int cls) {
  if (has_success) {
    VLOG(4) << op_name << " input: " << dim_in << ", output: " << dim_out
            << ", kernel dim: " << ksize[0] << ", " << ksize[1]
            << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
            << ", " << pads[3] << ", stride: " << strides[0] << ", "
            << strides[1]
            << ", global_pooling: " << (flag_global ? "global" : "false")
            << ", pooling_type: " << pooling_type
            << ", ceil_mode: " << (ceil_mode ? "true" : "false")
            << ", exclusive: " << (exclusive ? "true" : "false")
            << ", threads: " << th << ", power_mode: " << cls
            << " successed!!\n";
  } else {
    LOG(FATAL) << op_name << " input: " << dim_in << ", output: " << dim_out
               << ", kernel dim: " << ksize[0] << ", " << ksize[1]
               << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
               << ", " << pads[3] << ", stride: " << strides[0] << ", "
               << strides[1]
               << ", global_pooling: " << (flag_global ? "global" : "false")
               << ", pooling_type: " << pooling_type
               << ", ceil_mode: " << (ceil_mode ? "true" : "false")
               << ", exclusive: " << (exclusive ? "true" : "false")
               << ", threads: " << th << ", power_mode: " << cls
               << " failed!!\n";
  }
}
