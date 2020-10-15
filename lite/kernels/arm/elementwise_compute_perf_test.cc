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

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "lite/backends/arm/math/elementwise.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/elementwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <class T,
          void elementwise_op(const T* dinx, const T* diny, T* dout, int num)>
void do_element_perf(const char* func_name) {
  std::vector<int> Ns;
  std::vector<double> times;
  int N = 15;
  int N_Max = 16 * 1024 * 1024;

  timeval t_start;
  timeval t_end;
  double duration = 0;
  printf("Summary for [%s]\n", func_name);
  while (N <= N_Max) {
    const int REPEAT_TIME = std::max(5, 1024 * 1024 * 4 / N);
    duration = 0;
    for (int i = 0; i < REPEAT_TIME; ++i) {
      std::vector<T> x(N, 0);
      std::generate(x.begin(), x.end(), std::rand);
      std::vector<T> y(N, 0);
      std::generate(y.begin(), y.end(), std::rand);
      std::vector<T> z(N, 0);
      gettimeofday(&t_start, NULL);
      elementwise_op(x.data(), y.data(), z.data(), N);
      gettimeofday(&t_end, NULL);
      duration += ((t_end.tv_sec - t_start.tv_sec) * 1e6 +
                   (t_end.tv_usec - t_start.tv_usec)) /
                  REPEAT_TIME;
    }
    printf("N: %d us: %.14g \n", N, duration);
    Ns.push_back(N);
    times.push_back(duration);
    N *= 2;
  }
}

template <class T,
          void fast_braodcast_op(const T* dinx,
                                 const T* diny,
                                 T* dout,
                                 int batch,
                                 int channels,
                                 int num)>
void do_broadcast_perf(const char* func_name) {
  std::vector<int> Ns;
  std::vector<double> times;

  timeval t_start;
  timeval t_end;
  double duration = 0;
  printf("Summary for [%s]\n", func_name);
  for (auto batch : {1, 100, 300}) {
    for (auto channel : {1, 100, 300}) {
      for (auto num : {1, 100, 300}) {
        int N = batch * channel * num;
        const int REPEAT_TIME = std::max(5, 400 * 400 * 4 / N);
        duration = 0;
        for (int i = 0; i < REPEAT_TIME; ++i) {
          std::vector<T> x(batch * channel * num, 0);
          std::generate(x.begin(), x.end(), std::rand);
          std::vector<T> y(channel, 0);
          std::generate(y.begin(), y.end(), std::rand);
          std::vector<T> z(batch * channel * num, 0);
          gettimeofday(&t_start, NULL);
          fast_braodcast_op(x.data(), y.data(), z.data(), batch, channel, num);
          gettimeofday(&t_end, NULL);
          duration += ((t_end.tv_sec - t_start.tv_sec) * 1e6 +
                       (t_end.tv_usec - t_start.tv_usec)) /
                      REPEAT_TIME;
        }
        printf("N: %d us: %.14g \n", N, duration);
        Ns.push_back(N);
        times.push_back(duration);
      }
    }
  }
}

// These tests are disabled by this macro by default, To speed up CI testing
#ifdef ENABLE_ARM_PERF_TEST

TEST(elementwise_compute_perf, i32) {
  do_element_perf<int32_t, paddle::lite::arm::math::elementwise_add<int32_t>>(
      "elementwise_add");
  do_element_perf<int32_t, paddle::lite::arm::math::elementwise_sub<int32_t>>(
      "elementwise_sub");
  do_element_perf<int32_t, paddle::lite::arm::math::elementwise_mul<int32_t>>(
      "elementwise_mul");
  do_element_perf<int32_t, paddle::lite::arm::math::elementwise_div<int32_t>>(
      "elementwise_div");
}

TEST(elementwise_compute_perf, f32) {
  do_element_perf<float, paddle::lite::arm::math::elementwise_add<float>>(
      "elementwise_add");
  do_element_perf<float, paddle::lite::arm::math::elementwise_sub<float>>(
      "elementwise_sub");
  do_element_perf<float, paddle::lite::arm::math::elementwise_mul<float>>(
      "elementwise_mul");
  do_element_perf<float, paddle::lite::arm::math::elementwise_div<float>>(
      "elementwise_div");
  do_element_perf<float, paddle::lite::arm::math::elementwise_max<float>>(
      "elementwise_max");
}

TEST(elementwise_compute_relu_perf, f32) {
  do_element_perf<float, paddle::lite::arm::math::elementwise_add_relu<float>>(
      "elementwise_add_relu");
  do_element_perf<float, paddle::lite::arm::math::elementwise_sub_relu<float>>(
      "elementwise_sub_relu");
  do_element_perf<float, paddle::lite::arm::math::elementwise_mul_relu<float>>(
      "elementwise_mul_relu");
  do_element_perf<float, paddle::lite::arm::math::elementwise_div_relu<float>>(
      "elementwise_div_relu");
  do_element_perf<float, paddle::lite::arm::math::elementwise_max_relu<float>>(
      "elementwise_max_relu");
}

TEST(elementwise_compute_broadcast_perf, i32) {
  do_broadcast_perf<
      int32_t,
      paddle::lite::arm::math::elementwise_add_broadcast<int32_t>>(
      "elementwise_add_broadcast");
  do_broadcast_perf<
      int32_t,
      paddle::lite::arm::math::elementwise_sub_broadcast<int32_t>>(
      "elementwise_sub_broadcast");
  do_broadcast_perf<
      int32_t,
      paddle::lite::arm::math::elementwise_mul_broadcast<int32_t>>(
      "elementwise_mul_broadcast");
  do_broadcast_perf<
      int32_t,
      paddle::lite::arm::math::elementwise_div_broadcast<int32_t>>(
      "elementwise_div_broadcast");
}

TEST(elementwise_compute_broadcast_perf, f32) {
  do_broadcast_perf<float,
                    paddle::lite::arm::math::elementwise_add_broadcast<float>>(
      "elementwise_add_broadcast");
  do_broadcast_perf<float,
                    paddle::lite::arm::math::elementwise_sub_broadcast<float>>(
      "elementwise_sub_broadcast");
  do_broadcast_perf<float,
                    paddle::lite::arm::math::elementwise_mul_broadcast<float>>(
      "elementwise_mul_broadcast");
  do_broadcast_perf<float,
                    paddle::lite::arm::math::elementwise_div_broadcast<float>>(
      "elementwise_div_broadcast");
  do_broadcast_perf<float,
                    paddle::lite::arm::math::elementwise_max_broadcast<float>>(
      "elementwise_max_broadcast");
}

TEST(elementwise_compute_broadcast_relu_perf, f32) {
  do_broadcast_perf<
      float,
      paddle::lite::arm::math::elementwise_add_relu_broadcast<float>>(
      "elementwise_add_relu_broadcast");
  do_broadcast_perf<
      float,
      paddle::lite::arm::math::elementwise_sub_relu_broadcast<float>>(
      "elementwise_sub_relu_broadcast");
  do_broadcast_perf<
      float,
      paddle::lite::arm::math::elementwise_mul_relu_broadcast<float>>(
      "elementwise_mul_relu_broadcast");
  do_broadcast_perf<
      float,
      paddle::lite::arm::math::elementwise_div_relu_broadcast<float>>(
      "elementwise_div_relu_broadcast");
  do_broadcast_perf<
      float,
      paddle::lite::arm::math::elementwise_max_relu_broadcast<float>>(
      "elementwise_max_relu_broadcast");
}
#endif  // ENABLE_ARM_PERF_TEST
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_mul_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_max_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mod, kARM, kInt64, kNCHW, def);
