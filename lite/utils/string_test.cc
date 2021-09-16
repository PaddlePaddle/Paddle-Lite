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

#include "lite/utils/string.h"
#include <gtest/gtest.h>
#include <string>
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace utils {
namespace {

void check_stoi(const std::string& str, int base = 10) {
  int result;
  auto status = from_chars(str.data(), str.data() + str.size(), result, base);
  CHECK_EQ(result, std::stoi(str, 0, base));
  CHECK(status.ec == std::errc());
}

void check_stol(const std::string& str, int base = 10) {
  int64_t result;
  auto status = from_chars(str.data(), str.data() + str.size(), result, base);
  CHECK_EQ(result, std::stol(str, 0, base));
  CHECK(status.ec == std::errc());
}

void check_stof(const std::string& str) {
  float result;
  auto status = from_chars(str.data(), str.data() + str.size(), result);
  CHECK(std::abs(result - std::stof(str) < 0.0001));
  CHECK(status.ec == std::errc());
}

void check_stod(const std::string& str) {
  double result;
  auto status = from_chars(str.data(), str.data() + str.size(), result);
  CHECK(std::abs(result - std::stod(str) < 0.0001));
  CHECK(status.ec == std::errc());
}
}  // namespace

TEST(from_chars, test) {
  check_stoi("10");
  check_stoi("-128");
  check_stoi("A", 16);
  check_stol("100");
  check_stol("-128");
  check_stol("AA", 16);
  check_stof("10.10");
  check_stof("-10.10");
  check_stof("-3.1415926");
  check_stod("10.10");
  check_stod("-10.10");
  check_stod("-123.12345678");
}

TEST(StringView, Split) {
  const std::string str("conv2d/def/4/1/1");
  const std::vector<StringView> result = lite::SplitView(str, '/');
  CHECK_EQ(static_cast<std::string>(result[0]), "conv2d");
  CHECK_EQ(static_cast<std::string>(result[1]), "def");
  CHECK_EQ(static_cast<std::string>(result[2]), "4");
  CHECK_EQ(result[2].to_digit<int32_t>(), 4);
  CHECK_EQ(static_cast<std::string>(result[3]), "1");
  CHECK_EQ(result[3].to_digit<float>(), 1.0f);
  CHECK_EQ(static_cast<std::string>(result[4]), "1");
  CHECK_EQ(result[4].to_digit<uint32_t>(), 1u);
}

}  // namespace utils
}  // namespace lite
}  // namespace paddle
