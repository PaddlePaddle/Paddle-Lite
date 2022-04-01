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
#include <memory>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/kernels/arm/activation_compute.h"
#include "lite/kernels/arm/activation_extra_compute.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::DDim DDim;
typedef paddle::lite::operators::ActivationParam ActivationParam;
using paddle::lite::profile::Timer;

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "usage: " << argv[0] << "\n"
              << "  <batch_size>\n"
              << "  <input_channel>\n"
              << "  <input_height>\n"
              << "  <input_width>\n"
              << "  <act_type>\n"
              << "  <thread_num>\n"
              << "  <power_mode>\n"
              << "  <warmup_times>\n"
              << "  <repeats_times>" << std::endl;
    return 0;
  }

#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif

  int batch_size = atoi(argv[1]);
  int input_channel = atoi(argv[2]);
  int input_height = atoi(argv[3]);
  int input_width = atoi(argv[4]);
  int thread_num = atoi(argv[6]);
  int power_mode = atoi(argv[7]);
  int warmup = atoi(argv[8]);
  int repeats = atoi(argv[9]);
  int act_type = atoi(argv[5]);
  const float six = 6.f;
  const float leakey_relu_scale = 8.88f;

#ifdef LITE_WITH_ARM
  ActivationParam act_param;
  Tensor x, y;
  DDim dim_in = DDim({batch_size, input_channel, input_height, input_width});
  x.set_precision(PRECISION(kFloat));
  x.Resize(dim_in);
  paddle::lite::fill_tensor_rand(x, -1.f, 1.f);
  act_param.X = &x;
  act_param.active_type = (paddle::lite_api::ActivationType)act_type;
  act_param.has_active = true;

  if (act_type == 2) {
    act_param.Relu_clipped_coef = six;
  } else if (act_type == 4) {
    act_param.Leaky_relu_alpha = leakey_relu_scale;
  }

  act_param.Out = &y;
  act_param.Out->set_precision(PRECISION(kFloat));
  act_param.Out->Resize(dim_in);

  Timer t0;
  if (act_type == 1) {
    paddle::lite::kernels::arm::ReluCompute<PRECISION(kFloat)> act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 2) {
    paddle::lite::kernels::arm::Relu6Compute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 4) {
    paddle::lite::kernels::arm::LeakyReluCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 5) {
    paddle::lite::kernels::arm::SigmoidCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 6) {
    paddle::lite::kernels::arm::TanhCompute<PRECISION(kFloat)> act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 7) {
    paddle::lite::kernels::arm::SwishCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 8) {
    paddle::lite::kernels::arm::ExpCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 9) {
    paddle::lite::kernels::arm::AbsCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 10) {
    paddle::lite::kernels::arm::HardSwishCompute<PRECISION(kFloat)> act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 11) {
    paddle::lite::kernels::arm::ReciprocalCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  } else if (act_type == 12) {
    paddle::lite::kernels::arm::ThresholdedReluCompute act_compute;
    act_compute.SetParam(act_param);
    std::unique_ptr<paddle::lite::KernelContext> ctx1(
        new paddle::lite::KernelContext);
    auto& ctx = ctx1->As<paddle::lite::ARMContext>();
    ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(power_mode),
                   thread_num);
    act_compute.SetContext(std::move(ctx1));
    act_compute.PrepareForRun();
    // warm up
    for (int i = 0; i < warmup; ++i) {
      act_compute.Launch();
    }
    // compute
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      act_compute.Launch();
      t0.Stop();
    }
  }

  printf("Avg Latency is %f\n", t0.LapTimes().Avg());
  printf("Min Latency is %f\n", t0.LapTimes().Min());
  printf("Max Latency is %f\n", t0.LapTimes().Max());
#endif

  return 0;
}
