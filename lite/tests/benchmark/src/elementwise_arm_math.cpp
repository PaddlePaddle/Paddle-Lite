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

#include <vector>

#include "lite/backends/arm/math/elementwise.h"

static void fast_bcast_args(benchmark::internal::Benchmark* b) {
  b->ArgNames({"batch", "channel", "num"});
  for (auto batch : {1, 10, 30, 50, 100, 150}) {
    for (auto channel : {1, 10, 30, 50, 100, 150}) {
      for (auto num : {1, 10, 30, 50, 100, 150}) {
        b->Args({batch, channel, num});
      }
    }
  }
}

static void elementwise_args(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N"});
  int N = 15;
  int N_Max = 16 * 1024 * 1024;
  for (int i = N; i < N_Max; i *= 16) {
    b->Arg(i);
  }
}

template <class T,
          void elementwise_op(const T* dinx, const T* diny, T* dout, int num)>
void do_element_perf(const benchmark::State& state_in, int complexity_factor) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State& state = const_cast<benchmark::State&>(state_in);

  int N = state.range(0);

  std::vector<T> x(N, 0);
  std::generate(x.begin(), x.end(), std::rand);
  std::vector<T> y(N, 0);
  std::generate(y.begin(), y.end(), std::rand);
  std::vector<T> z(N, 0);

  for (auto _ : state) {
    elementwise_op(x.data(), y.data(), z.data(), N);
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * N * complexity_factor,
                         benchmark::Counter::kIsRate);
}

template <class T,
          void fast_braodcast_op(const T* dinx,
                                 const T* diny,
                                 T* dout,
                                 int batch,
                                 int channels,
                                 int num)>
void do_broadcast_perf(const benchmark::State& state_in,
                       int complexity_factor) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State& state = const_cast<benchmark::State&>(state_in);
  int batch = state.range(0);
  int channel = state.range(1);
  int num = state.range(2);
  std::vector<T> x(batch * channel * num, 0);
  std::generate(x.begin(), x.end(), std::rand);
  std::vector<T> y(channel, 0);
  std::generate(y.begin(), y.end(), std::rand);
  std::vector<T> z(batch * channel * num, 0);

  for (auto _ : state) {
    fast_braodcast_op(x.data(), y.data(), z.data(), batch, channel, num);
  }

  state.counters["OPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * batch * channel * num * complexity_factor,
      benchmark::Counter::kIsRate);
}

#define BENCHMARK_ELEMENTWISE(elementwise_op, data_t, complexity_factor)  \
  static constexpr auto elementwise_op##_##data_t =                       \
      do_element_perf<data_t,                                             \
                      paddle::lite::arm::math::elementwise_op<data_t>>;   \
  BENCHMARK_CAPTURE(elementwise_op##_##data_t, nomral, complexity_factor) \
      ->Apply(elementwise_args)                                           \
      ->UseRealTime();

#define BENCHMARK_ELEMENTWISE_FAST_BCAST(                                 \
    elementwise_op, data_t, complexity_factor)                            \
  static constexpr auto elementwise_op##_##data_t =                       \
      do_broadcast_perf<data_t,                                           \
                        paddle::lite::arm::math::elementwise_op<data_t>>; \
  BENCHMARK_CAPTURE(                                                      \
      elementwise_op##_##data_t, fast_broadcast, complexity_factor)       \
      ->Apply(fast_bcast_args)                                            \
      ->UseRealTime();

BENCHMARK_ELEMENTWISE(elementwise_add, int32_t, 1);
BENCHMARK_ELEMENTWISE(elementwise_sub, int32_t, 1);
BENCHMARK_ELEMENTWISE(elementwise_mul, int32_t, 1);
BENCHMARK_ELEMENTWISE(elementwise_div, int32_t, 1);

BENCHMARK_ELEMENTWISE(elementwise_add, float, 1);
BENCHMARK_ELEMENTWISE(elementwise_sub, float, 1);
BENCHMARK_ELEMENTWISE(elementwise_mul, float, 1);
BENCHMARK_ELEMENTWISE(elementwise_div, float, 1);
BENCHMARK_ELEMENTWISE(elementwise_max, float, 1);

BENCHMARK_ELEMENTWISE(elementwise_add_relu, float, 2);
BENCHMARK_ELEMENTWISE(elementwise_sub_relu, float, 2);
BENCHMARK_ELEMENTWISE(elementwise_mul_relu, float, 2);
BENCHMARK_ELEMENTWISE(elementwise_div_relu, float, 2);
BENCHMARK_ELEMENTWISE(elementwise_max_relu, float, 2);

BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_add_broadcast, int32_t, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_sub_broadcast, int32_t, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_mul_broadcast, int32_t, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_div_broadcast, int32_t, 1)

BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_add_broadcast, float, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_sub_broadcast, float, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_mul_broadcast, float, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_div_broadcast, float, 1)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_max_broadcast, float, 1)

BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_add_relu_broadcast, float, 2)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_sub_relu_broadcast, float, 2)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_mul_relu_broadcast, float, 2)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_div_relu_broadcast, float, 2)
BENCHMARK_ELEMENTWISE_FAST_BCAST(elementwise_max_relu_broadcast, float, 2)

BENCHMARK_MAIN();
