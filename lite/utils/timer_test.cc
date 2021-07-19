// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/utils/timer.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

TEST(timer, basic) {
  paddle::lite::Timer timer;
  for (float i = 0.f; i < 100.f; i += 10.f) {
    timer.Start();
    timer.SleepInMs(i);
    float elapsed_time_ms = timer.Stop();
    timer.Print();
    CHECK_EQ(elapsed_time_ms, i);
  }

  const float ms = 1000.f timer.Start();
  timer.SleepInMs(ms);
  float elapsed_time_ms = timer.Stop();
  timer.Print();
  CHECK_EQ(elapsed_time_ms, ms);
}

}  // namespace lite
}  // namespace paddle
