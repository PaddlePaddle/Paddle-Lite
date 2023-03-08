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
#include <stdio.h>
#include <unistd.h>

#include <random>
#include <vector>

#include "lite/backends/arm/math/gemm_s8.h"
#include "lite/core/context.h"
#include "lite/tests/benchmark/src/gemm_configs.h"

template <class CValueT>
static void test_gemm_s8(const benchmark::State &state_in,
                         bool istranA,
                         bool isTransB,
                         bool has_bias,
                         bool has_relu) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State &state = const_cast<benchmark::State &>(state_in);

  const int m = state.range(0);
  const int n = state.range(1);
  const int k = state.range(2);

  using paddle::lite::DDim;
  using paddle::lite::Tensor;
  Tensor x, y, z;
  Tensor bias, scale;
  DDim dim_x = DDim({m, k});
  DDim dim_y = DDim({k, n});
  DDim dim_z = DDim({m, n});
  DDim dim_bias_scale = DDim({m});
  x.set_precision(PRECISION(kInt8));
  x.Resize(dim_x);
  y.set_precision(PRECISION(kInt8));
  y.Resize(dim_y);
  if (std::is_same<CValueT, float>::value) {
    z.set_precision(PRECISION(kFloat));
  } else {
    z.set_precision(PRECISION(kInt8));
  }
  z.Resize(dim_z);
  bias.set_precision(PRECISION(kFloat));
  bias.Resize(dim_bias_scale);
  scale.set_precision(PRECISION(kFloat));
  scale.Resize(dim_bias_scale);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             std::numeric_limits<int8_t>::min(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::generate(x.mutable_data<int8_t>(),
                x.mutable_data<int8_t>() + x.numel(),
                std::ref(i8rng));
  std::generate(y.mutable_data<int8_t>(),
                y.mutable_data<int8_t>() + y.numel(),
                std::ref(i8rng));
  std::generate(bias.mutable_data<float>(),
                bias.mutable_data<float>() + bias.numel(),
                std::ref(f32rng));
  std::generate(scale.mutable_data<float>(),
                scale.mutable_data<float>() + scale.numel(),
                std::ref(f32rng));
  z.mutable_data<float>();  // pre alloc

  paddle::lite::operators::ActivationParam act_param;
  act_param.has_active = has_relu;
  if (has_relu) {
    act_param.active_type = (paddle::lite_api::ActivationType)1;
  }

  auto ctx1 = paddle::lite::ContextScheduler::Global().NewContext(
      paddle::lite_api::TargetType::kARM);
  auto &ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(
                     paddle::lite_api::PowerMode::LITE_POWER_HIGH),
                 1);

  for (auto _ : state) {
    paddle::lite::arm::math::gemm_s8(istranA,
                                     isTransB,
                                     false,
                                     m,
                                     n,
                                     k,
                                     x.data<int8_t>(),
                                     y.data<int8_t>(),
                                     z.mutable_data<CValueT>(),
                                     bias.data<float>(),
                                     has_bias,
                                     scale.data<float>(),
                                     act_param,
                                     &ctx);
  }
  float op_ratio = 2;

  if (has_bias) {
    ++op_ratio;
  }
  if (has_relu) {
    ++op_ratio;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * m * n * k,
                         benchmark::Counter::kIsRate);
  state.counters["RATIO"] = benchmark::Counter(op_ratio / 2);
}

static void int8_out_s8_gemm_A_nt_B_nt_no_bias_no_relu(
    const benchmark::State &state, const char *net) {
  test_gemm_s8<int8_t>(state, false, false, false, false);
}

BENCHMARK_GEMM(int8_out_s8_gemm_A_nt_B_nt_no_bias_no_relu)

BENCHMARK_MAIN();
