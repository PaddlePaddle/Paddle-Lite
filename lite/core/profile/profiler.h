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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/profile/timer.h"

namespace paddle {
namespace lite {
namespace profile {

enum class Type {
  kUnk = 0,
  kCreate,
  kDispatch,
};

extern std::map<Type, std::string> TypeStr;

struct TimeInfo {
  float avg;
  float min;
  float max;
};

struct OpCharacter {
  TargetType target;
  std::string op_type{std::string("N/A")};
  std::string kernel_name{std::string("N/A")};
  std::string remark{std::string("N/A")};
};

class StatisUnit final {
 public:
  explicit StatisUnit(const OpCharacter& ch);
  lite::profile::Timer* Timer(Type type);
  const OpCharacter& Character() const { return character; }

 protected:
  std::unique_ptr<lite::profile::Timer> create_t;
  std::unique_ptr<lite::profile::Timer> dispatch_t;
  OpCharacter character;
};

class Profiler final {
 public:
  Profiler() = default;
  explicit Profiler(const std::string& name) : name_(name) {}
  int NewTimer(const OpCharacter& ch);
  void StartTiming(Type type, const int index, KernelContext* ctx);
  float StopTiming(Type type, const int index, KernelContext* ctx);
  std::string Summary(Type type, bool concise = true, size_t warm_up = 10);

 private:
  std::string name_{std::string("N/A")};
  std::vector<StatisUnit> units_;
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle
