// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <iostream>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/kernels/arm/conv_compute.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::DDim DDim;
typedef paddle::lite::operators::ActivationParam ActivationParam;

using paddle::lite::profile::Timer;
using paddle::lite_api::PrecisionType;

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

template <PrecisionType Ptype, PrecisionType OutType>
void test_conv(const DDim& input_dims,
               const DDim& weight_dims,
               const int group,
               const std::vector<int>& strides,
               const std::vector<int>& pads,
               const std::vector<int>& dilas,
               const bool flag_bias,
               const int flag_act,
               const int thread_num,
               const int power_mode,
               const int warmup,
               const int repeats,
               const float leakey_relu_scale = 8.88f) {
  ConvParam param;
  Tensor x, f, y;
  Tensor bias;
  param.x = &x;
  param.x->set_precision(Ptype);
  param.filter = &f;
  param.filter->Resize(weight_dims);
  param.filter->set_precision(Ptype);
  if (flag_bias) {
    param.bias = &bias;
    param.bias->Resize({weight_dims[0]});
    param.bias->set_precision(PRECISION(kFloat));
  }
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.groups = group;
  const float six = 6.f;

  if (Ptype == PRECISION(kInt8)) {
    std::vector<float> scale_in{1.f / 127};
    std::vector<float> scale_out(1, weight_dims.count(1, 4) / 127.f);
    if (flag_act == 2) {
      scale_out[0] = six / 127.f;
    } else if (flag_act == 4) {
      if (std::abs(leakey_relu_scale) > 1) {
        scale_out[0] *= std::abs(leakey_relu_scale);
      }
    }
    std::vector<float> scale_w(weight_dims[0], 1.f / 127);
    param.input_scale = scale_in[0];
    param.output_scale = scale_out[0];
    param.weight_scale = scale_w;
  }

  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type = (paddle::lite_api::ActivationType)
        flag_act;  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      param.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = leakey_relu_scale;
    }
    param.activation_param = act_param;
  }

  param.output = &y;
  param.output->set_precision(OutType);

  paddle::lite::fill_tensor_rand(*param.filter, -1.f, 1.f);
  if (flag_bias) {
    paddle::lite::fill_tensor_rand(*param.bias, -1.f, 1.f);
  }

  paddle::lite::kernels::arm::ConvCompute<Ptype, OutType> conv;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                 thread_num);

  param.x->Resize(input_dims);
  DDim dim_out = compute_out_dim(input_dims, param);
  param.output->Resize(dim_out);
  conv.SetParam(param);
  conv.SetContext(std::move(ctx1));
  conv.PrepareForRun();
  paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);

  // warm up
  for (int i = 0; i < warmup; ++i) {
    conv.Launch();
  }
  // compute
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    conv.Launch();
    t0.Stop();
  }
  printf("Avg Latency is %f\n", t0.LapTimes().Avg());
  printf("Min Latency is %f\n", t0.LapTimes().Min());
  printf("Max Latency is %f\n", t0.LapTimes().Max());
}

int main(int argc, char** argv) {
  if (argc != 23) {
    std::cerr << "usage: " << argv[0] << "\n"
              << "  <batch_size>\n"
              << "  <input_channel>\n"
              << "  <input_height>\n"
              << "  <input_width>\n"
              << "  <output_channel>\n"
              << "  <group_size>\n"
              << "  <kernel_size>\n"
              << "  <pad_top>\n"
              << "  <pad_bottom>\n"
              << "  <pad_left>\n"
              << "  <pad_right>\n"
              << "  <stride_h>\n"
              << "  <stride_w>\n"
              << "  <dilation_h>\n"
              << "  <dilation_w>\n"
              << "  <flag_bias>\n"
              << "  <flag_act>\n"
              << "  <dtype>\n"
              << "  <thread_num>\n"
              << "  <power_mode>\n"
              << "  <warmup_times>\n"
              << "  <repeats_times>\n"
              << std::endl;
    return 0;
  }
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  int batch_size = atoi(argv[1]);
  int input_channel = atoi(argv[2]);
  int input_height = atoi(argv[3]);
  int input_width = atoi(argv[4]);
  int output_channel = atoi(argv[5]);
  int group_size = atoi(argv[6]);
  int kernel_size = atoi(argv[7]);
  int pad_top = atoi(argv[8]);
  int pad_bottom = atoi(argv[9]);
  int pad_left = atoi(argv[10]);
  int pad_right = atoi(argv[11]);
  int stride_h = atoi(argv[12]);
  int stride_w = atoi(argv[13]);
  int dilation_h = atoi(argv[14]);
  int dilation_w = atoi(argv[15]);
  int flag_bias = atoi(argv[16]);
  int flag_act = atoi(argv[17]);
  int dtype = atoi(argv[18]);
  int thread_num = atoi(argv[19]);
  int power_mode = atoi(argv[20]);
  int warmup = atoi(argv[21]);
  int repeats = atoi(argv[22]);

  DDim weight_dims(
      {output_channel, input_channel / group_size, kernel_size, kernel_size});
  DDim input_dims({batch_size, input_channel, input_height, input_width});
  switch (dtype) {
    case 0:
      test_conv<PRECISION(kFloat), PRECISION(kFloat)>(
          input_dims,
          weight_dims,
          group_size,
          {stride_h, stride_w},
          {pad_top, pad_bottom, pad_left, pad_right},
          {dilation_h, dilation_w},
          flag_bias,
          flag_act,
          thread_num,
          power_mode,
          warmup,
          repeats);
      break;
    case 1:
      test_conv<PRECISION(kInt8), PRECISION(kFloat)>(
          input_dims,
          weight_dims,
          group_size,
          {stride_h, stride_w},
          {pad_top, pad_bottom, pad_left, pad_right},
          {dilation_h, dilation_w},
          flag_bias,
          flag_act,
          thread_num,
          power_mode,
          warmup,
          repeats);
      break;
    case 2:
      test_conv<PRECISION(kInt8), PRECISION(kInt8)>(
          input_dims,
          weight_dims,
          group_size,
          {stride_h, stride_w},
          {pad_top, pad_bottom, pad_left, pad_right},
          {dilation_h, dilation_w},
          flag_bias,
          flag_act,
          thread_num,
          power_mode,
          warmup,
          repeats);
      break;
    default:
      test_conv<PRECISION(kFloat), PRECISION(kFloat)>(
          input_dims,
          weight_dims,
          group_size,
          {stride_h, stride_w},
          {pad_top, pad_bottom, pad_left, pad_right},
          {dilation_h, dilation_w},
          flag_bias,
          flag_act,
          thread_num,
          power_mode,
          warmup,
          repeats);
  }

  return 0;
}
