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

namespace {
  auto op_comp = [](const OpCharacter& c1, const OpCharacter& c2) {
    return (c1.target < c2.target) || (c1.op_type < c2.op_type) ||
           (c1.kernel_name < c2.kernel_name) || (c1.remark < c2.remark);
  };
}

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
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  units_[index].timer->Start(ctx);
}

float Profiler::StopTiming(const int index, KernelContext* ctx) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  return units_[index].timer->Stop(ctx);
}

std::string Profiler::Summary(bool concise) {
  using std::setw;
  using std::left;
  using std::fixed;
  STL::stringstream ss;
  std::string title;
  // Title.
  if (concise) {
    ss << "Timing cycle = " << units_.front().timer->LapTimes().Size() << std::endl;
    ss << "===== Concise Profiler Summary: " << name_ << " =====" << std::endl;
  } else {
    ss << "===== Detailed Profiler Summary: " << name_ << " =====" << std::endl;
  }
  ss << setw(25) << left << "Operator Type" \
     << " " << setw(40) << left << "Kernel Name"   \
     << " " << setw(12) << left << "Remark"        \
     << " " << setw(12) << left << "Avg (ms)"      \
     << " " << setw(12) << left << "Min (ms)"      \
     << " " << setw(12) << left << "Max (ms)"      \
     << " " << setw(12) << left << "Last (ms)"     \
     << std::endl;
  // Profile information.
  if (concise) {
    std::map<OpCharacter, TimeInfo, decltype(op_comp)> summary(op_comp);
    for (auto& unit : units_) {
      auto ch = summary.find(unit.character);
      if (ch != summary.end()) {
        ch->second.avg += unit.timer->LapTimes().Avg();
        ch->second.min += unit.timer->LapTimes().Min();
        ch->second.max += unit.timer->LapTimes().Max();
      } else {
        TimeInfo info({unit.timer->LapTimes().Avg(),
                       unit.timer->LapTimes().Min(),
                       unit.timer->LapTimes().Max()});
        summary.insert({unit.character, info});
      }
    }
    for (const auto& item : summary) {
      // clang-format off
      ss << setw(25) << left << fixed << item.first.op_type             \
         << " " << setw(40) << left << fixed << item.first.kernel_name  \
         << " " << setw(12) << left << fixed << item.first.remark       \
         << " " << setw(12) << left << fixed << item.second.avg         \
         << " " << setw(12) << left << fixed << item.second.min         \
         << " " << setw(12) << left << fixed << item.second.max         \
         << " " << std::endl;
      // clang-format on
    }
  } else {
    for (auto& unit : units_) {
      // clang-format off
      ss << setw(25) << left << fixed << unit.character.op_type              \
         << " " << setw(40) << left << fixed << unit.character.kernel_name    \
         << " " << setw(12) << left << fixed << unit.character.remark         \
         << " " << setw(12) << left << fixed << unit.timer->LapTimes().Avg()  \
         << " " << setw(12) << left << fixed << unit.timer->LapTimes().Min()  \
         << " " << setw(12) << left << fixed << unit.timer->LapTimes().Max()  \
         << " " << setw(12) << left << fixed << unit.timer->LapTimes().Last() \
         << std::endl;
      // clang-format on
    }
  }
  return ss.str();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
