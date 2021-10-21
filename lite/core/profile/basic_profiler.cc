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
#include <map>
#include <string>
#if defined(TARGET_IOS)
const char* FLAGS_time_profile_file = "time_profile.txt";
const char* FLAGS_time_profile_summary_file = "time_profile_summary.txt";
const char* FLAGS_time_profile_unit = "ms";
#else
DEFINE_string(time_profile_file,
              "time_profile.txt",
              "Lite time profile information dump file");

DEFINE_string(time_profile_summary_file,
              "time_profile_summary.txt",
              "Lite time profile summary information dump file");

DEFINE_string(time_profile_unit,
              "ms",
              "Unit of time in profile infomation, ms or us");
#endif
namespace paddle {
namespace lite {
namespace profile {

static std::string GetTimeUnit() {
  auto time_unit = FLAGS_time_profile_unit;
  if (time_unit != "ms" && time_unit != "us") {
    LOG(FATAL) << "Profile time unit only support ms or us now";
  }
  return time_unit;
}

const int BasicTimer::data_w = 10;
const int BasicTimer::name_w = 15;

void BasicTimer::Start(const std::string& timer_key) {
  TimerInfo& timer_info = timer_infos_[timer_key];
  timer_info.timer_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

void BasicTimer::Stop(const std::string& timer_key) {
  if (timer_infos_.find(timer_key) == timer_infos_.end()) {
    LOG(FATAL) << "Error: Can't found timer key [" << timer_key << "] for "
               << key_;
  }
  TimerInfo& timer_info = timer_infos_[timer_key];
  auto duration = static_cast<
      uint64_t>(  // timer unit: microsecond, 1second = 1e6 microsecond
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count() -
      timer_info.timer_);
  Log(&timer_info, duration);
}

void BasicTimer::SetCustomInfo(const std::string& key,
                               const std::string& value) {
  if (custom_infos_.find(key) != custom_infos_.end()) {
    LOG(FATAL) << "Error: Custom Info for key [" << key
               << "] can't be overwritten";
  }
  custom_infos_[key] = value;
}

std::string BasicTimer::GetCustomInfo(const std::string& key) const {
  auto iter = custom_infos_.find(key);
  if (iter == custom_infos_.end()) {
    LOG(FATAL) << "Error: Custom Info for key [" << key << "] can't be found";
  }
  return iter->second;
}

const TimerInfo& BasicTimer::GetTimerInfo(const std::string& key) const {
  auto iter = timer_infos_.find(key);
  if (iter == timer_infos_.end()) {
    LOG(FATAL) << "Error: Timer Info for key [" << key << "] can't be found";
  }
  return iter->second;
}

void BasicTimer::SetWarmup(int warmup_times) {
  CHECK_GE(warmup_times, 0) << "warmup times must >= 0";
  warmup_ = warmup_times;
}

void BasicTimer::Log(TimerInfo* timer_info, uint64_t timespan) {
  if (warmup_ > 0) {
    --warmup_;
    return;
  }
  CHECK(timer_info);
  timer_info->count_++;
  timer_info->total_ += timespan;
  timer_info->max_ = (std::max)(timer_info->max_, timespan);
  timer_info->min_ = (std::min)(timer_info->min_, timespan);
}

std::string BasicTimer::basic_repr_header() {
  auto time_unit = GetTimeUnit();
  STL::stringstream ss;
  // clang-format off
  ss << "op"         << "\t"
     << "kernel"     << "\t"
     << "k_average(" << time_unit << ")\t"
     << "k_min("     << time_unit << ")\t"
     << "k_max("     << time_unit << ")\t"
     << "i_average(" << time_unit << ")\t"
     << "i_min("     << time_unit << ")\t"
     << "i_max("     << time_unit << ")\t"
     << "count"      << "\t"
     << "op_info";
  // clang-format on
  return ss.str();
}

std::string BasicTimer::basic_repr() const {
  auto& kernel_timer_info = GetTimerInfo("kernel");
  auto& inst_timer_info = GetTimerInfo("instruction");
  float time_unit_factor = 1.;
  if (GetTimeUnit() == "ms") {
    time_unit_factor = 1000.;
  }
  STL::stringstream ss;
  // clang-format off
  ss << GetCustomInfo("op_type")                    << "\t"
     << key()                                       << "\t"
     << kernel_timer_info.Ave() / time_unit_factor  << "\t"
     << kernel_timer_info.Min() / time_unit_factor  << "\t"
     << kernel_timer_info.Max() / time_unit_factor  << "\t"
     << inst_timer_info.Ave()   / time_unit_factor  << "\t"
     << inst_timer_info.Min()   / time_unit_factor  << "\t"
     << inst_timer_info.Max()   / time_unit_factor  << "\t"
     << inst_timer_info.Count()                     << "\t"
     << GetCustomInfo("op_info");
  // clang-format on
  return ss.str();
}

template class BasicProfiler<BasicTimer>;

template <typename TimerT>
std::string BasicProfiler<TimerT>::summary_repr_header() const {
  auto time_unit = GetTimeUnit();
  STL::stringstream ss;
  // clang-format off
  ss << "op"          << "\t"
     << "average("    << time_unit << ")\t"
     << "min("        << time_unit << ")\t"
     << "max("        << time_unit << ")\t"
     << "op_time("    << time_unit << ")\t"
     << "total_time(" << time_unit << ")\t"
     << "precent"     << "\t"
     << "count";
  // clang-format on
  return ss.str();
}

template <typename TimerT>
std::string BasicProfiler<TimerT>::summary_repr() const {
  std::map<std::string, TimerInfo> op_summary;
  uint64_t total{0};

  for (const auto& rcd : records_) {
    // We use kernel run time here
    auto kernel_timer = rcd.GetTimerInfo("kernel");
    auto op_type = rcd.GetCustomInfo("op_type");
    auto& op_timer = op_summary[op_type];

    total += kernel_timer.total_;
    op_timer.total_ += kernel_timer.total_;
    op_timer.max_ = (std::max)(kernel_timer.max_, op_timer.max_);
    op_timer.min_ = (std::min)(kernel_timer.min_, op_timer.min_);
    op_timer.count_ += kernel_timer.count_;
  }

  float time_unit_factor = 1.;
  if (GetTimeUnit() == "ms") {
    time_unit_factor = 1000.;
  }
  STL::stringstream ss;
  for (auto& iter : op_summary) {
    auto& op_timer = iter.second;
    // clang-format off
    ss << iter.first                             << "\t"
       << op_timer.Ave()   / time_unit_factor    << "\t"
       << op_timer.Min()   / time_unit_factor    << "\t"
       << op_timer.Max()   / time_unit_factor    << "\t"
       << op_timer.Total() / time_unit_factor    << "\t"
       << total            / time_unit_factor    << "\t"
       << (op_timer.Total() * 1. / total * 100)  << "%\t"
       << op_timer.Count()                       << "\t"
       << "\n";
    // clang-format on
  }
  return ss.str();
}

template <typename TimerT>
BasicProfiler<TimerT>::~BasicProfiler() {
  LOG(INFO) << "Basic Profile dumps:";
  auto b_repr = TimerT::basic_repr_header() + "\n" + basic_repr();
  LOG(INFO) << "\n" + b_repr;

  // Dump to file
  std::ofstream basic_ostream(FLAGS_time_profile_file);
  CHECK(basic_ostream.is_open()) << "Open " << FLAGS_time_profile_file
                                 << " failed";
  basic_ostream.write(b_repr.c_str(), b_repr.size());
  basic_ostream.close();

  LOG(INFO) << "Summary Profile dumps:";
  auto s_repr = summary_repr_header() + "\n" + summary_repr();
  LOG(INFO) << "\n" + s_repr;

  // Dump to file
  std::ofstream summary_ostream(FLAGS_time_profile_summary_file);
  CHECK(summary_ostream.is_open()) << "Open " << FLAGS_time_profile_summary_file
                                   << " failed";
  summary_ostream.write(s_repr.c_str(), s_repr.size());
  summary_ostream.close();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
