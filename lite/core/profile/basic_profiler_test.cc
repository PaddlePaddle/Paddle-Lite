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

#include "lite/core/profile/basic_profiler.h"
#include <gtest/gtest.h>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace profile {

TEST(basic_record, init) {
  BasicTimer timer;
  timer.SetKey("hello");
}

TEST(basic_profile, real_latency) {
  auto profile_id = profile::BasicProfiler<profile::BasicTimer>::Global()
                        .NewRcd("test0")
                        .id();
  auto& profiler =
      *BasicProfiler<profile::BasicTimer>::Global().mutable_record(profile_id);
  // Set op info
  profiler.SetCustomInfo("op_type", "fc");
  profiler.SetCustomInfo("op_info", "size:5x6");

  profile::ProfileBlock x(profile_id, "instruction");
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  profile::ProfileBlock y(profile_id, "kernel");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
