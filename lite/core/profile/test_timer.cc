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

}  // namespace profile
}  // namespace lite
}  // namespace paddle
