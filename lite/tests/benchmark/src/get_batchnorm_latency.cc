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
#include "lite/kernels/arm/batch_norm_compute.h"
#include "lite/operators/op_params.h"

typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::kernels::arm::BatchNormCompute<float, PRECISION(kFloat)>
    BatchNormCompute;
using paddle::lite::profile::Timer;

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cerr << "usage: " << argv[0] << "\n"
              << "  <batch_size>\n"
              << "  <input_channel>\n"
              << "  <input_height>\n"
              << "  <input_width>\n"
              << "  <epsilon>\n"
              << "  <momentum>\n"
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
  float epsilon = atof(argv[5]);
  float momentum = atof(argv[6]);
  int thread_num = atoi(argv[7]);
  int power_mode = atoi(argv[8]);
  int warmup = atoi(argv[9]);
  int repeats = atoi(argv[10]);

#ifdef LITE_WITH_ARM
  Tensor x;
  Tensor scale;
  Tensor bias;
  Tensor mean;
  Tensor variance;
  Tensor y;
  Tensor mean_out;
  Tensor variance_out;
  Tensor saved_mean;
  Tensor saved_variance;

  std::vector<int64_t> in_out_shape = {
      batch_size, input_channel, input_height, input_width};
  x.Resize(in_out_shape);
  scale.Resize({input_channel});
  bias.Resize({input_channel});
  mean.Resize({input_channel});
  variance.Resize({input_channel});
  y.Resize(in_out_shape);
  mean_out.Resize({input_channel});
  variance_out.Resize({input_channel});
  saved_mean.Resize({input_channel});
  saved_variance.Resize({input_channel});
  // initialize the data of input tensors
  auto* x_data = x.mutable_data<float>();
  auto* scale_data = scale.mutable_data<float>();
  auto* bias_data = bias.mutable_data<float>();
  auto* mean_data = mean.mutable_data<float>();
  auto* variance_data = variance.mutable_data<float>();
  for (int i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i % 64);
  }
  for (int i = 0; i < scale.dims().production(); i++) {
    scale_data[i] = static_cast<float>(i) * 0.01f + 0.03f;
  }
  for (int i = 0; i < bias.dims().production(); i++) {
    bias_data[i] = static_cast<float>(i) * 0.065f + 0.1f;
  }
  for (int i = 0; i < mean.dims().production(); i++) {
    mean_data[i] = static_cast<float>(i) * 0.0565f;
  }
  for (int i = 0; i < variance.dims().production(); i++) {
    variance_data[i] = static_cast<float>(i) * 2.08f + 1.5f;
  }

  // prepare kernel params and run
  BatchNormCompute batch_norm;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                 thread_num);
  batch_norm.SetContext(std::move(ctx1));

  paddle::lite::operators::BatchNormParam param;
  param.x = &x;
  param.scale = &scale;
  param.bias = &bias;
  param.mean = &mean;
  param.variance = &variance;
  param.is_test = false;
  param.use_global_stats = true;
  param.epsilon = epsilon;
  param.momentum = momentum;
  param.data_layout = DATALAYOUT(kNCHW);
  param.y = &y;
  param.mean_out = &mean_out;
  param.variance_out = &variance_out;
  param.saved_mean = &saved_mean;
  param.saved_variance = &saved_variance;
  batch_norm.SetParam(param);

  // warm up
  for (int i = 0; i < warmup; ++i) {
    batch_norm.Launch();
  }
  // compute
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    batch_norm.Launch();
    t0.Stop();
  }
  printf("Avg Latency is %f\n", t0.LapTimes().Avg());
  printf("Min Latency is %f\n", t0.LapTimes().Min());
  printf("Max Latency is %f\n", t0.LapTimes().Max());
#endif

  return 0;
}
