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

#include "lite/core/profile/profiler.h"
#include <map>
#include <string>
#include <utility>

namespace paddle {
namespace lite {
namespace profile {

int Profiler::NewTimer(const OpCharacter& ch) {
  StatisUnit unit;
  unit.character = ch;
  if (ch.target == TargetType::kCUDA) {
#ifdef LITE_WITH_CUDA
    unit.timer.reset(new DeviceTimer<TargetType::kCUDA>());
#else
    LOG(ERROR) << "The timer type specified as cuda is uninitialized, so the "
                  "default x86 timer is used instead.";
#endif
  } else {
    unit.timer.reset(new DeviceTimer<TargetType::kHost>());
  }
  units_.push_back(std::move(unit));
  return units_.size() - 1;
}

void Profiler::StartTiming(const int index, KernelContext* ctx) {
  units_[index].timer->Start(ctx);
}

float Profiler::StopTiming(const int index, KernelContext* ctx) {
  return units_[index].timer->Stop(ctx);
}

std::string Profiler::Summary(bool concise) {
  STL::stringstream ss;
  auto cout_title = [&ss](const std::string& title, const std::string& name) {
    // clang-format off
    ss << "===== " << title << ": " << name << " =====" << std::endl;
    ss << std::setw(25) << std::left << "Operator Type" \
       << std::setw(40) << std::left << "Kernel Name"   \
       << std::setw(10) << std::left << "Remark"        \
       << std::setw(10) << std::left << "Avg (ms)"      \
       << std::setw(10) << std::left << "Min (ms)"      \
       << std::setw(10) << std::left << "Max (ms)"      \
       << std::endl;
    // clang-format on
  };
  if (concise) {
    auto op_comp = [](const OpCharacter& c1, const OpCharacter& c2) {
      return (c1.target < c2.target) || (c1.op_type < c2.op_type) ||
             (c1.kernel_name < c2.kernel_name) || (c1.remark < c2.remark);
    };
    std::map<OpCharacter, TimeInfo, decltype(op_comp)> summary(op_comp);
    for (auto& unit : units_) {
      auto ch = summary.find(unit.character);
      if (ch != summary.end()) {
        ch->second.avg += unit.timer->LapsTime().Avg();
        ch->second.min += unit.timer->LapsTime().Min();
        ch->second.max += unit.timer->LapsTime().Max();
      } else {
        TimeInfo info({unit.timer->LapsTime().Avg(),
                       unit.timer->LapsTime().Min(),
                       unit.timer->LapsTime().Max()});
        summary.insert({unit.character, info});
      }
    }
    cout_title("Concise Profiler Summary", name_);
    for (const auto& item : summary) {
      // clang-format off
      ss << std::setw(25) << std::left << item.first.op_type      \
         << std::setw(40) << std::left << item.first.kernel_name  \
         << std::setw(10) << std::left << item.first.remark       \
         << std::setw(10) << std::left << item.second.avg         \
         << std::setw(10) << std::left << item.second.min         \
         << std::setw(10) << std::left << item.second.max         \
         << std::endl;
      // clang-format on
    }
  } else {
    cout_title("Detailed Profiler Summary", name_);
    for (auto& unit : units_) {
      // clang-format off
      ss << std::setw(25) << std::left << unit.character.op_type        \
         << std::setw(40) << std::left << unit.character.kernel_name    \
         << std::setw(10) << std::left << unit.character.remark         \
         << std::setw(10) << std::left << unit.timer->LapsTime().Avg()  \
         << std::setw(10) << std::left << unit.timer->LapsTime().Min()  \
         << std::setw(10) << std::left << unit.timer->LapsTime().Max()  \
         << std::endl;
      // clang-format on
    }
  }
  return ss.str();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
