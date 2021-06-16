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
#include <iomanip>
#include <map>
#include <string>
#include <utility>

namespace paddle {
namespace lite {
namespace profile {

namespace {
auto op_comp = [](const OpCharacter& c1, const OpCharacter& c2) {
  // compare for unique key of map
  return (c1.kernel_name + c1.kernel_func_name <
          c2.kernel_name + c2.kernel_func_name);
};
}  // namespace

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

OpCharacter* Profiler::GetOpCharacter(const size_t index) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  return &units_[index].Character();
}

void Profiler::StartTiming(Type type, const int index, KernelContext* ctx) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
  units_[index].Timer(type)->Start(ctx);
}

void Profiler::StopTiming(Type type, const int index, KernelContext* ctx) {
  CHECK_LT(index, units_.size())
      << "The timer index in the profiler is out of range.";
#ifdef LITE_WITH_OPENCL
  units_[index].Timer(type)->CLStop(units_[index].character.op_type,
                                    units_[index].character.io_duration,
                                    units_[index].character.cl_event);
#endif
  units_[index].Timer(type)->Stop(ctx);
}

int Profiler::GetKernelFuncCalledTimes(const std::string& op_type,
                                       const std::string& kernel_attr,
                                       const std::string& kernel_func_name) {
  int count = 0;
  for (size_t i = 0; i < units_.size(); ++i) {
    if ((units_[i].character.kernel_func_name == kernel_func_name) &&
        (units_[i].character.kernel_attr == kernel_attr) &&
        (units_[i].character.op_type == op_type)) {
      ++count;
    }
  }
  return count;
}

float Profiler::GetKernelFuncSummaryGOPs(const std::string& op_type,
                                         const std::string& kernel_attr,
                                         const std::string& kernel_func_name) {
  float GOPs = 0;
  for (size_t i = 0; i < units_.size(); ++i) {
    if ((units_[i].character.kernel_func_name == kernel_func_name) &&
        (units_[i].character.kernel_attr == kernel_attr) &&
        (units_[i].character.op_type == op_type)) {
      GOPs += units_[i].character.macs;
    }
  }
  return GOPs * 1e-9f;
}

std::string Profiler::Summary(Type type, bool concise, size_t w) {
  using std::setw;
  using std::left;
  using std::fixed;
  using std::setprecision;
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
  ss << setw(20) << left << "OperatorType"
     << " " << setw(30) << left << "KerneAttr(Place)"
     << " " << setw(24) << left << "KernelFuncName";
  if (!concise) {
    ss << " " << setw(26) << left << "Remark"
       << " " << setw(15) << left << "InDim"
       << " " << setw(15) << left << "FilterDim"
       << " " << setw(15) << left << "OutDim";
  }
  ss << " " << setw(7) << left << "Avg(ms)"
     << " " << setw(7) << left << "Min(ms)"
     << " " << setw(7) << left << "Max(ms)";
  if (!concise) {
    ss << " " << setw(7) << left << "Last(ms)";
  }
  ss << " " << setw(7) << left << "Avg(%)"
     << " " << setw(7) << left << "GOPs";
  if (!concise) {
    ss << " " << setw(7) << left << "GOPS";
  }
  if (concise) {
    ss << " " << setw(11) << left << "CalledTimes";
  }
#ifdef LITE_WITH_OPENCL
  ss << " " << setw(9) << left << "clAvg(ms)"
     << " " << setw(9) << left << "clMin(ms)"
     << " " << setw(9) << left << "clMax(ms)"
     << " " << setw(9) << left << "clAvg(%)";
  if (!concise) {
    ss << " " << setw(12) << left << "GlobalWorkSize"
       << " " << setw(12) << left << "LocalWorkSize";
  }
#endif
  ss << std::endl;

  // Profile information.
  if (concise) {
    std::map<OpCharacter, TimeInfo, decltype(op_comp)> summary(op_comp);
    for (auto& unit : units_) {
      auto ch = summary.find(unit.Character());
      if (ch != summary.end()) {
        ch->second.avg += unit.Timer(type)->LapTimes().Avg(w);
        ch->second.min += unit.Timer(type)->LapTimes().Min(w);
        ch->second.max += unit.Timer(type)->LapTimes().Max(w);
#ifdef LITE_WITH_OPENCL
        ch->second.cl_avg += unit.Timer(type)->CLLapTimes().Avg(w);
        ch->second.cl_min += unit.Timer(type)->CLLapTimes().Min(w);
        ch->second.cl_max += unit.Timer(type)->CLLapTimes().Max(w);
#endif
      } else {
        TimeInfo info;
        info.avg = unit.Timer(type)->LapTimes().Avg(w);
        info.min = unit.Timer(type)->LapTimes().Min(w);
        info.max = unit.Timer(type)->LapTimes().Max(w);
#ifdef LITE_WITH_OPENCL
        info.cl_avg = unit.Timer(type)->CLLapTimes().Avg(w);
        info.cl_min = unit.Timer(type)->CLLapTimes().Min(w);
        info.cl_max = unit.Timer(type)->CLLapTimes().Max(w);
#endif
        summary.insert({unit.Character(), info});
      }
    }

    // compute total time
    float total = 0.0;
    for (const auto& item : summary) {
      total += item.second.avg;
    }
#ifdef LITE_WITH_OPENCL
    float cl_total = 0.0;
    for (const auto& item : summary) {
      cl_total += item.second.cl_avg;
    }
#endif

    for (const auto& item : summary) {
      float percent = 0;
      if (total > 0) {
        percent = 100 * (item.second.avg / total);
      }
      // clang-format off
      ss << setw(20) << left << fixed << item.first.op_type
         << " " << setw(30) << left << fixed << item.first.kernel_attr
         << " " << setw(24) << left << fixed << item.first.kernel_func_name
         << " " << setw(7) << left << fixed << setprecision(3)
         << item.second.avg
         << " " << setw(7) << left << fixed << setprecision(3)
         << item.second.min
         << " " << setw(7) << left << fixed << setprecision(3)
         << item.second.max
         << " " << setprecision(2) << percent << "%   "
         << " " << setw(7) << left << fixed << setprecision(3)
         << GetKernelFuncSummaryGOPs(item.first.op_type,
                                     item.first.kernel_attr,
                                     item.first.kernel_func_name)
         << " " << setw(11) << left << fixed
         << GetKernelFuncCalledTimes(item.first.op_type,
                                     item.first.kernel_attr,
                                     item.first.kernel_func_name);
#ifdef LITE_WITH_OPENCL
      float cl_percent = 0;
      if (cl_total > 0) {
        cl_percent = 100 * (item.second.cl_avg / cl_total);
      }
      ss << " " << setw(9) << left << fixed << setprecision(3)
         << item.second.cl_avg
         << " " << setw(9) << left << fixed << setprecision(3)
         << item.second.cl_min
         << " " << setw(9) << left << fixed << setprecision(3)
         << item.second.cl_max
         << " " << left << fixed << setprecision(2) << cl_percent << "%   ";
#endif
      ss << std::endl;
      // clang-format on
    }
  } else {
    float total = 0.0;
    for (auto& unit : units_) {
      const auto& times = unit.Timer(type)->LapTimes();
      total += times.Avg(w);
    }
#ifdef LITE_WITH_OPENCL
    float cl_total = 0.0;
    for (auto& unit : units_) {
      const auto& cl_times = unit.Timer(type)->CLLapTimes();
      cl_total += cl_times.Avg(w);
    }
#endif
    for (auto& unit : units_) {
      const auto& times = unit.Timer(type)->LapTimes();
      float run = times.Avg(w);
      float percent = 0;
      if (total > 0) {
        percent = 100 * (run / total);
      }

#ifdef LITE_WITH_OPENCL
      const auto& cl_times = unit.Timer(type)->CLLapTimes();
      float cl_run = cl_times.Avg(w);
      float cl_percent = 0;
      if (cl_total > 0) {
        cl_percent = 100 * (cl_run / cl_total);
      }
#endif

      // clang-format off
      ss << setw(20) << left << fixed << unit.Character().op_type
         << " " << setw(30) << left << fixed << unit.Character().kernel_attr
         << " " << setw(24) << left << fixed
         << unit.Character().kernel_func_name
         << " " << setw(26) << left << fixed << unit.Character().remark
         << " " << setw(15) << left << fixed << unit.Character().input_shape
         << " " << setw(15) << left << fixed << unit.Character().filter_shape
         << " " << setw(15) << left << fixed << unit.Character().output_shape
         << " " << setw(7) << left << fixed << setprecision(3) << times.Avg(w)
         << " " << setw(7) << left << fixed << setprecision(3) << times.Min(w)
         << " " << setw(7) << left << fixed << setprecision(3) << times.Max(w)
         << " " << setw(7) << left << fixed << setprecision(3) << times.Last(w)
         << " " << left << setprecision(2) << percent << "%   "
         << " " << setw(7) << left << fixed << setprecision(3)
                << 1e-9f * unit.Character().macs
         << " " << setw(7) << left << fixed << setprecision(2)
                << 1e-6f * unit.Character().macs / times.Avg(w);
// clang-format on
#ifdef LITE_WITH_OPENCL
      ss << " " << setw(9) << left << fixed << setprecision(3)
         << cl_times.Avg(w) << " " << setw(9) << left << fixed
         << setprecision(3) << cl_times.Min(w) << " " << setw(9) << left
         << fixed << setprecision(3) << cl_times.Max(w) << " " << left
         << setprecision(2) << cl_percent << "%   "
         << " " << setw(12) << left << fixed
         << unit.Character().global_work_size << " " << setw(12) << left
         << fixed << unit.Character().local_work_size;
#endif
      ss << std::endl;
    }
  }
  return ss.str();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
