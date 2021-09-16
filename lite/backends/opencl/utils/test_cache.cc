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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/backends/opencl/utils/cache.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace fbs {
namespace opencl {

TEST(OpenCLCache, cache) {
  const std::map<std::string, std::vector<std::vector<uint8_t>>> map{
      {"a", {{1, 2}, {3, 4}}}, {"b", {{5, 6}, {7, 8}}},
  };
  Cache cache_0{map};
  std::vector<uint8_t> buffer;
  cache_0.CopyDataToBuffer(&buffer);

  Cache cache_1{buffer};
  CHECK(map == cache_1.GetBinaryMap())
      << "Cache read and write are not equivalent, the test failed.";
}

TEST(OpenCLTunedCache, tuned_cache) {
  const std::map<std::string, std::vector<int>> map{
      {"a", {1, 1, 1}}, {"b", {1, 1, 2}},
  };
  TuneCache cache_0{map};
  std::vector<int> buffer;
  cache_0.CopyDataToBuffer(&buffer);

  TuneCache cache_1{buffer};
  CHECK(map == cache_1.GetBinaryMap())
      << "Cache read and write are not equivalent, the test failed.";
}

}  // namespace opencl
}  // namespace fbs
}  // namespace lite
}  // namespace paddle
