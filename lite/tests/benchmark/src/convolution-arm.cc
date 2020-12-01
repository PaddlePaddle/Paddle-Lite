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
#include <unistd.h>

#include <random>

#include "lite/kernels/arm/conv_compute.h"
#include "lite/tests/benchmark/src/convolution_configs.h"

template <class Tin,
          class Tout,
          paddle::lite::PrecisionType Ptype,
          paddle::lite::PrecisionType OutType>
void bench_conv(const benchmark::State& state_in, const char* net) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State& state = const_cast<benchmark::State&>(state_in);

  const int64_t batch_size = state.range(0);
  const int64_t input_height = state.range(1);
  const int64_t input_width = state.range(2);
  const int64_t kernel_height = state.range(3);
  const int64_t kernel_width = state.range(4);
  const int64_t padding_height = state.range(5);
  const int64_t padding_width = state.range(6);
  const int64_t subsampling = state.range(7);
  const int stride_v = subsampling;
  const int64_t dilation = state.range(8);
  const int64_t groups = state.range(9);
  const int64_t group_input_channels = state.range(10);
  const int64_t group_output_channels = state.range(11);

  const int64_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const int64_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const int64_t output_height =
      (input_height + padding_height - effective_kernel_height) / subsampling +
      1;
  const int64_t output_width =
      (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(-10, 10), std::ref(rng));
  using paddle::lite::DDim;
  using paddle::lite::Tensor;

  Tensor x, filter, bias, output;
  x.Resize(DDim(
      {batch_size, groups * group_input_channels, input_height, input_width}));
  std::generate(x.mutable_data<Tin>(),
                x.mutable_data<Tin>() + x.numel(),
                std::ref(input_rng));
  filter.Resize(DDim({groups * group_output_channels,
                      group_input_channels,
                      kernel_height,
                      kernel_width}));
  std::generate(filter.mutable_data<Tin>(),
                filter.mutable_data<Tin>() + filter.numel(),
                std::ref(input_rng));
  bias.Resize(DDim({groups * group_output_channels}));
  std::generate(bias.mutable_data<float>(),
                bias.mutable_data<float>() + bias.numel(),
                std::ref(input_rng));
  output.Resize(DDim({batch_size,
                      groups * group_output_channels,
                      output_height,
                      output_width}));
  output.mutable_data<Tout>();

  paddle::lite::kernels::arm::ConvCompute<Ptype, OutType> conv_compute;
  paddle::lite::operators::ConvParam param;
  param.x = &x;
  param.bias = &bias;
  param.filter = &filter;
  param.output = &output;

  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  auto pd = std::make_shared<std::vector<int>>();
  pd->push_back(padding_top);
  pd->push_back(padding_bottom);
  pd->push_back(padding_left);
  pd->push_back(padding_right);
  param.paddings = pd;

  param.strides = std::vector<int>{stride_v, stride_v};
  param.groups = groups;
  auto dl = std::make_shared<std::vector<int>>();
  dl->push_back(dilation);
  dl->push_back(dilation);
  param.dilations = dl;

  if (std::is_same<int8_t, Tin>::value) {
    param.enable_int8 = true;
    param.weight_scale = std::vector<float>({1});
  }

  conv_compute.SetParam(param);

  auto ctx1 = paddle::lite::ContextScheduler::Global().NewContext(
      paddle::lite_api::TargetType::kARM);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(
                     paddle::lite_api::PowerMode::LITE_POWER_HIGH),
                 1);

  conv_compute.SetContext(std::move(ctx1));
  conv_compute.PrepareForRun();

  for (auto _ : state) {
    conv_compute.Launch();
  }

  state.counters["OPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * 2 * batch_size * output_height *
          output_width * groups * group_input_channels * group_output_channels *
          kernel_height * kernel_width,
      benchmark::Counter::kIsRate);
}
constexpr static auto f32_conv =
    bench_conv<float, float, PRECISION(kFloat), PRECISION(kFloat)>;
BENCHMARK_CONVOLUTION(f32_conv)

constexpr static auto int8_conv =
    bench_conv<int8_t, int8_t, PRECISION(kInt8), PRECISION(kInt8)>;
BENCHMARK_CONVOLUTION(int8_conv)

BENCHMARK_MAIN();
