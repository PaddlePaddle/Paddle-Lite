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
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif

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

DEFINE_int32(out_channel, 32, "output channel");
DEFINE_int32(group, 1, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h0, 1, "pad top");
DEFINE_int32(pad_h1, 1, "pad bottom");
DEFINE_int32(pad_w0, 1, "pad left");
DEFINE_int32(pad_w1, 1, "pad right");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_int32(flag_act,
             0,
             "do activation");  // 0-no act, 1-relu, 2-relu6, 4-leakyrelu
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_bool(flag_bias, true, "with bias");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::operators::ActivationParam ActivationParam;

using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  DDim dim_out = dim_in;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + pad_top + pad_bottom - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + pad_left + pad_right - kernel_exten) / stride_w + 1;
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

#define VLOG_PRINT_DIFF(a, b) \
  VLOG(4) << "compare result, max diff: " << a << ", max ratio: " << b;

#define VLOG_PRINT_GOPS(input_dim, dim_out, t0, gops)                    \
  VLOG(4) << "conv fp16: input shape: " << input_dim << ", output shape" \
          << dim_out << ",running time, avg: " << t0.LapTimes().Avg()    \
          << ", min time: " << t0.LapTimes().Min()                       \
          << ", total GOPS: " << 1e-9 * gops                             \
          << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()    \
          << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

#define VLOG_DIFF_INFO(basic_ptr, saber_ptr, ptr, size, width) \
  LOG(WARNING) << "basic result";                              \
  print_tensor(basic_ptr, size, width);                        \
  LOG(WARNING) << "lite result";                               \
  print_tensor(saber_ptr, size, width);                        \
  LOG(WARNING) << "diff result";                               \
  print_tensor(ptr, size, width);

#define VLOG_FINAL_INFO(input_dim, dim_out)                                \
  << ", output: " << dim_out << ", weight dim: " << weight_dim             \
  << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2] << ", "    \
  << pads[3] << ", stride: " << strides[0] << ", " << strides[1]           \
  << ", dila_: " << dilas[0] << ", " << dilas[1] << ", group: " << group   \
  << ", bias: " << (flag_bias ? "true" : "false") << ", act: " << flag_act \
  << ", threads: " << th

#define VLOG_FAILED_INFO(input_dim, dim_out)                  \
  LOG(FATAL) << "test fp16 conv: input: "                     \
             << input_dim VLOG_FINAL_INFO(input_dim, dim_out) \
             << ", power_mode: " << cls << " failed!!\n";

#define VLOG_SUCCESSED_INFO(input_dim, dim_out)            \
  VLOG(4) << "test fp16 conv: input: "                     \
          << input_dim VLOG_FINAL_INFO(input_dim, dim_out) \
          << ", power_mode: " << cls << " successed!!\n";
