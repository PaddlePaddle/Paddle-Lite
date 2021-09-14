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
#include <chrono>  // NOLINT
#include <thread>  // NOLINT
#include "lite/core/context.h"
#include "lite/core/profile/profiler.h"
#include "lite/core/profile/timer.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace profile {

TEST(timer, real_latency) {
  Timer timer;

  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.Stop();

  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  timer.Stop();

  LOG(INFO) << "LapTimes().Avg() = " << timer.LapTimes().Avg();
}

#ifdef LITE_WITH_CUDA
TEST(gpu_timer, real_latency) {
  DeviceTimer<TargetType::kCUDA> timer;
  KernelContext ctx;
  cudaStream_t exec_stream;
  cudaStreamCreate(&exec_stream);
  (&ctx.As<CUDAContext>())->SetExecStream(exec_stream);

  timer.Start(&ctx);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.Stop(&ctx);

  (&timer)->Start(&ctx);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  timer.Stop(&ctx);

  LOG(INFO) << "LapTimes().Avg() = " << timer.LapTimes().Avg();
}

TEST(profiler, real_latency) {
  KernelContext ctx;
  cudaStream_t exec_stream;
  cudaStreamCreate(&exec_stream);
  (&ctx.As<CUDAContext>())->SetExecStream(exec_stream);

  Profiler profiler("name");
  profile::OpCharacter ch;
  ch.target = TargetType::kCUDA;
  ch.op_type = "operator/1";
  ch.kernel_name = "kernel/1";
  int idx = profiler.NewTimer(ch);
  profiler.StartTiming(Type::kDispatch, idx, &ctx);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  profiler.StopTiming(Type::kDispatch, idx, &ctx);
  std::cout << profiler.Summary(Type::kDispatch);
}
#endif

}  // namespace profile
}  // namespace lite
}  // namespace paddle
