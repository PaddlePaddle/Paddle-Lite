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
#include "lite/kernels/arm/fc_compute.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::DDim DDim;
typedef paddle::lite::operators::FcParam FcParam;
using paddle::lite::profile::Timer;
using paddle::lite_api::PrecisionType;

template <PrecisionType Ptype, PrecisionType OutType>
void test_fc(const int m,
             const int n,
             const int k,
             const bool has_bias,
             const int thread_num,
             const int power_mode,
             const int warmup,
             const int repeats) {
  FcParam param;
  Tensor x, y, bias, w;
  param.input = &x;
  param.input->set_precision(Ptype);
  param.input->Resize({m, k});
  param.w = &w;
  param.w->set_precision(Ptype);
  param.w->Resize({k, n});
  if (has_bias) {
    param.bias = &bias;
    param.bias->set_precision(Ptype);
    param.bias->Resize({1, n});
  } else {
    param.bias = nullptr;
  }
  param.output = &y;
  param.output->set_precision(OutType);
  param.output->Resize({m, n});

  param.in_num_col_dims = 1;
  param.in_mat_dims = param.input->dims();

  paddle::lite::kernels::arm::FcCompute<Ptype, OutType> fc_compute;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                 thread_num);
  // set param and context
  fc_compute.SetParam(param);
  fc_compute.SetContext(std::move(ctx1));
  // prepare for run
  fc_compute.PrepareForRun();
  paddle::lite::fill_tensor_rand(*param.input, -1.f, 1.f);
  paddle::lite::fill_tensor_rand(*param.w, -1.f, 1.f);

  if (has_bias) {
    paddle::lite::fill_tensor_rand(*param.bias, -1.f, 1.f);
  }
  // warm up
  for (int i = 0; i < warmup; ++i) {
    fc_compute.Launch();
  }
  // compute
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    fc_compute.Launch();
    t0.Stop();
  }

  printf("Avg Latency is %f\n", t0.LapTimes().Avg());
  printf("Min Latency is %f\n", t0.LapTimes().Min());
  printf("Max Latency is %f\n", t0.LapTimes().Max());
}

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "usage: " << argv[0] << "\n"
              << " <m>\n"
              << " <n>\n"
              << " <k>\n"
              << " <has_bias>\n"
              << " <dtype>\n"
              << " <thread_num>\n"
              << " <power_mode>\n"
              << " <warmup_times>\n"
              << " <repeats_times>\n"
              << std::endl;
    return 0;
  }
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int k = atoi(argv[3]);
  bool has_bias = atoi(argv[4]) == 0 ? false : true;
  int dtype = argv[5] == "int8_int8" ? 2 : argv[5] == "float_int8"
                                               ? 1
                                               : argv[5] == "float" ? 0 : 0;
  int thread_num = atoi(argv[6]);
  int power_mode = atoi(argv[7]);
  int warmup = atoi(argv[8]);
  int repeats = atoi(argv[9]);

  switch (dtype) {
    case 0:
      test_fc<PRECISION(kFloat), PRECISION(kFloat)>(
          m, n, k, has_bias, thread_num, power_mode, warmup, repeats);
      break;
    case 1:
      test_fc<PRECISION(kInt8), PRECISION(kFloat)>(
          m, n, k, has_bias, thread_num, power_mode, warmup, repeats);
      break;
    case 2:
      test_fc<PRECISION(kInt8), PRECISION(kInt8)>(
          m, n, k, has_bias, thread_num, power_mode, warmup, repeats);
      break;
    default:
      test_fc<PRECISION(kFloat), PRECISION(kFloat)>(
          m, n, k, has_bias, thread_num, power_mode, warmup, repeats);
      break;
  }

  return 0;
}
