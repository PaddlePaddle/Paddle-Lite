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

#include <benchmark/benchmark.h>

#include <random>

#include "lite/tests/benchmark/src/gemm_configs.h"

#include "lite/core/context.h"
#include "lite/kernels/arm/matmul_compute.h"

static void LiteGEMMBench(const benchmark::State &state_in) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State &state = const_cast<benchmark::State &>(state_in);

  const int mc = state.range(0);
  const int nc = state.range(1);
  const int kc = state.range(2);

  using paddle::lite::DDim;
  using paddle::lite::Tensor;

  paddle::lite::kernels::arm::MatMulCompute matmul_compute;
  Tensor x, y, z;
  DDim dim_x = DDim({mc, kc});
  DDim dim_y = DDim({kc, nc});
  DDim dim_z = DDim({mc, nc});

  x.set_precision(PRECISION(kFloat));
  x.Resize(dim_x);
  y.set_precision(PRECISION(kFloat));
  y.Resize(dim_y);
  z.set_precision(PRECISION(kFloat));
  z.Resize(dim_z);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::generate(x.mutable_data<float>(),
                x.mutable_data<float>() + x.numel(),
                std::ref(f32rng));
  std::generate(y.mutable_data<float>(),
                y.mutable_data<float>() + y.numel(),
                std::ref(f32rng));
  z.mutable_data<float>();  // pre alloc

  paddle::lite::operators::MatMulParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &z;
  matmul_compute.SetParam(param);

  auto ctx1 = paddle::lite::ContextScheduler::Global().NewContext(
      paddle::lite_api::TargetType::kARM);
  auto &ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(
                     paddle::lite_api::PowerMode::LITE_POWER_HIGH),
                 1);

  matmul_compute.SetContext(std::move(ctx1));
  matmul_compute.PrepareForRun();

  for (int i = 0; i < 2; ++i) {
    matmul_compute.Launch();
  }

  for (auto _ : state) {
    matmul_compute.Launch();
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

static void paddle_f32_gemm(const benchmark::State &state, const char *net) {
  LiteGEMMBench(state);
}

BENCHMARK_GEMM(paddle_f32_gemm)

BENCHMARK_MAIN();
