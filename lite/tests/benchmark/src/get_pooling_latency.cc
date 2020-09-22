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
#include "lite/kernels/arm/pool_compute.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::DDim DDim;
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

int main(int argc, char** argv) {
  if (argc != 20) {
    std::cerr << "usage: " << argv[0] << "\n"
              << "  <batch_size>\n"
              << "  <input_channel>\n"
              << "  <input_height>\n"
              << "  <input_width>\n"
              << "  <kernel_size>\n"
              << "  <stride_size>\n"
              << "  <pad_size>\n"
              << "  <exclusive>\n"
              << "  <pooling_type>\n"
              << "  <ceil_mode>\n"
              << "  <flag_global>\n"
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
  int stride_h = atoi(argv[5]);
  int stride_w = atoi(argv[6]);
  int pad_top = atoi(argv[7]);
  int pad_bottom = atoi(argv[8]);
  int pad_left = atoi(argv[9]);
  int pad_right = atoi(argv[10]);
  int kernel_size = atoi(argv[11]);
  bool ceil_mode = argv[12] == 0 ? false : true;
  bool flag_global = argv[13] == 0 ? false : true;
  bool exclusive = atoi(argv[14]) == 0 ? false : true;
  std::string pooling_type = atoi(argv[15]) == 0 ? "max" : "avg";
  int thread_num = atoi(argv[16]);
  int power_mode = atoi(argv[17]);
  int warmup = atoi(argv[18]);
  int repeats = atoi(argv[19]);

#ifdef LITE_WITH_ARM
  PoolParam param;
  Tensor x, y;
  param.x = &x;
  param.x->set_precision(PRECISION(kFloat));
  param.ksize = {kernel_size, kernel_size};
  param.strides = {stride_h, stride_w};
  param.paddings = std::make_shared<std::vector<int>>(
      std::vector<int>{pad_top, pad_bottom, pad_left, pad_right});
  param.ceil_mode = ceil_mode;
  param.global_pooling = flag_global;
  param.pooling_type = pooling_type;
  param.exclusive = exclusive;
  param.adaptive = false;
  param.use_quantizer = false;
  param.output = &y;
  param.output->set_precision(PRECISION(kFloat));

  paddle::lite::kernels::arm::PoolCompute pool;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                 thread_num);
  // set param and context
  pool.SetParam(param);
  pool.SetContext(std::move(ctx1));
  // prepare for run
  pool.PrepareForRun();
  DDim dim_in = DDim({batch_size, input_channel, input_height, input_width});
  DDim dim_out = compute_out_dim(dim_in, param);

  param.x->Resize(dim_in);
  param.output->Resize(dim_out);

  paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);
  // warm up
  for (int i = 0; i < warmup; ++i) {
    pool.Launch();
  }
  // compute
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    pool.Launch();
    t0.Stop();
  }

  printf("Avg Latency is %f\n", t0.LapTimes().Avg());
  printf("Min Latency is %f\n", t0.LapTimes().Min());
  printf("Max Latency is %f\n", t0.LapTimes().Max());
#endif

  return 0;
}
