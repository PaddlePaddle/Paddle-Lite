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

std::map<Type, std::string> TypeStr{
    {Type::kUnk, "Unknown"},
    {Type::kCreate, "Create"},
    {Type::kDispatch, "Dispatch"},
};

StatisUnit::StatisUnit(const OpCharacter& ch) : character(ch) {
  create_t.reset(new DeviceTimer<TargetType::kHost>());
  if (ch.target == TargetType::kCUDA) {
#ifdef LITE_WITH_CUDA
    dispatch_t.reset(new DeviceTimer<TargetType::kCUDA>());
#else
    LOG(ERROR) << "The timer type specified as cuda is uninitialized, so the "
                  "default x86 timer is used instead.";
#endif
  } else {
    dispatch_t.reset(new DeviceTimer<TargetType::kHost>());
  }
}

lite::profile::Timer* StatisUnit::Timer(Type type) {
  if (type == Type::kCreate) {
    return create_t.get();
  } else if (type == Type::kDispatch) {
    return dispatch_t.get();
  }
  LOG(FATAL) << "Timer cannot be returned for unknown platforms.";
  return nullptr;
}

int Profiler::NewTimer(const OpCharacter& ch) {
  StatisUnit unit(ch);
  units_.push_back(std::move(unit));
  return units_.size() - 1;
}

void Profiler::StartTiming(Type type, const int index, KernelContext* ctx) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  units_[index].Timer(type)->Start(ctx);
}

float Profiler::StopTiming(Type type, const int index, KernelContext* ctx) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  return units_[index].Timer(type)->Stop(ctx);
}

std::string Profiler::Summary(Type type, bool concise, size_t w) {
  using std::setw;
  using std::left;
  using std::fixed;
  STL::stringstream ss;
  std::string title;
  // Title.
  if (concise) {
    ss << "Timing cycle = " << units_.front().Timer(type)->LapTimes().Size()
       << std::endl;
    ss << "===== Concise " << TypeStr.find(type)->second
       << " Profiler Summary: " << name_ << ", Exclude " << w
       << " warm-ups =====" << std::endl;
  } else {
    ss << "===== Detailed " << TypeStr.find(type)->second
       << " Profiler Summary: " << name_ << ", Exclude " << w
       << " warm-ups =====" << std::endl;
  }
  ss << setw(25) << left << "Operator Type"
     << " " << setw(40) << left << "Kernel Name"
     << " " << setw(12) << left << "Remark"
     << " " << setw(12) << left << "Avg (ms)"
     << " " << setw(12) << left << "Min (ms)"
     << " " << setw(12) << left << "Max (ms)"
     << " " << setw(12) << left << "Last (ms)"
     << " " << setw(12) << left << "Percent (%)" << std::endl;
  // Profile information.
  if (concise) {
    std::map<OpCharacter, TimeInfo, decltype(op_comp)> summary(op_comp);
    for (auto& unit : units_) {
      auto ch = summary.find(unit.Character());
      if (ch != summary.end()) {
        ch->second.avg += unit.Timer(type)->LapTimes().Avg(w);
        ch->second.min += unit.Timer(type)->LapTimes().Min(w);
        ch->second.max += unit.Timer(type)->LapTimes().Max(w);
      } else {
        TimeInfo info({unit.Timer(type)->LapTimes().Avg(w),
                       unit.Timer(type)->LapTimes().Min(w),
                       unit.Timer(type)->LapTimes().Max(w)});
        summary.insert({unit.Character(), info});
      }
    }
    // compute total time
    float total = 0.0;
    for (const auto& item : summary) {
      total += item.second.avg;
    }
    for (const auto& item : summary) {
      float percent = 0;
      if (total > 0) {
        percent = 100 * (item.second.avg / total);
      }
      // clang-format off
      ss << setw(25) << left << fixed << item.first.op_type             \
         << " " << setw(40) << left << fixed << item.first.kernel_name  \
         << " " << setw(12) << left << fixed << item.first.remark       \
         << " " << setw(12) << left << fixed << item.second.avg         \
         << " " << setw(12) << left << fixed << item.second.min         \
         << " " << setw(12) << left << fixed << item.second.max         \
         << " " << setw(12) << left << fixed << percent << "%"          \
         << " " << std::endl;
      // clang-format on
    }
  } else {
    float total = 0.0;
    for (auto& unit : units_) {
      const auto& times = unit.Timer(type)->LapTimes();
      total += times.Avg(w);
    }
    for (auto& unit : units_) {
      const auto& times = unit.Timer(type)->LapTimes();
      float run = times.Avg(w);
      float percent = 0;
      if (total > 0) {
        percent = 100 * (run / total);
      }
      // clang-format off
      ss << setw(25) << left << fixed << unit.Character().op_type            \
         << " " << setw(40) << left << fixed << unit.Character().kernel_name \
         << " " << setw(12) << left << fixed << unit.Character().remark      \
         << " " << setw(12) << left << fixed << times.Avg(w)                 \
         << " " << setw(12) << left << fixed << times.Min(w)                 \
         << " " << setw(12) << left << fixed << times.Max(w)                 \
         << " " << setw(12) << left << fixed << times.Last(w)                \
          << " " << setw(12) << left << fixed << percent << "%"              \
         << std::endl;
      // clang-format on
    }
  }
  return ss.str();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
