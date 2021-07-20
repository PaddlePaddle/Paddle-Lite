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

#if !defined(_WIN32)
#include <sys/time.h>
#include <sys/types.h>
#else
extern struct timeval;
static int gettimeofday(struct timeval* tp, void* tzp) {
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  tp->tv_sec = now.QuadPart / freq.QuadPart;
  tp->tv_usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
  // uint64_t elapsed_time = sec * 1000000 + usec;

  return (0);
}
#endif // !_WIN32

TEST(timer, basic) {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };

  const float scale = 0.1f;
  paddle::lite::Timer timer;
  for (float ms = 0.f; ms < 100.f; ms += 25.f) {
    timer.Start();
    timer.SleepInMs(ms);
    float elapsed_time_ms = timer.Stop();
    timer.Print();
    EXPECT_NEAR(elapsed_time_ms, ms, ms * scale);

    auto start = GetCurrentUS();
    timer.SleepInMs(ms);
    auto end = GetCurrentUS();
    float base = (end - start) * 1e-3;
    LOG(INFO) << "Base time: " << ms
              << "  gettimeofday: " << base
              << "  Timer: " << elapsed_time_ms;
    EXPECT_NEAR(elapsed_time_ms, base, base * scale);
  }

  const float ms = 123.f;
  timer.Start();
  timer.SleepInMs(ms);
  float elapsed_time_ms = timer.Stop();
  timer.Print();
  EXPECT_NEAR(elapsed_time_ms, ms, ms * scale);
}

}  // namespace lite
}  // namespace paddle
