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

/*
 * This file implements BasicProfile, a profiler that helps to profile the basic
 * CPU execution. It can display the min, max, average lantency of the execution
 * of each kernel.
 */
#pragma once
#include <gflags/gflags.h>
#include <time.h>
#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/utils/cp_logging.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace profile {

struct TimerInfo {
  uint64_t total_{};
  uint64_t count_{};
  uint32_t max_{std::numeric_limits<uint32_t>::min()};
  uint32_t min_{std::numeric_limits<uint32_t>::max()};
  uint64_t timer_{};

  double ave() const { return total_ * 1. / count_; }
  double max() const { return max_; }
  double min() const { return min_; }
  uint64_t total() const { return total_; }
  uint64_t count() const { return count_; }
};

/* Base class of all the profile records */
template <typename ChildT>
class TimerBase {
 public:
  void Start(const std::string& key) { self()->Start(key); }
  void Stop(const std::string& key) { self()->Stop(key); }
  void Log(TimerInfo* timer_info, uint32_t x) {
    return self()->Log(timer_info, x);
  }
  std::string basic_repr() const { return const_self()->basic_repr(); }

  void SetId(int id) { self()->SetId(id); }
  void SetKey(const std::string& key) { self()->SetKey(key); }

  int id() const { return const_self()->id(); }

 protected:
  ChildT* self() { return reinterpret_cast<ChildT*>(this); }
  const ChildT* const_self() const {
    return reinterpret_cast<const ChildT*>(this);
  }
};

class BasicTimer : TimerBase<BasicTimer> {
  int id_{-1};
  std::string key_;
  std::map<std::string, TimerInfo> timer_infos_;
  std::map<std::string, std::string> custom_infos_;

  // TODO(Superjomn) make static
  static const int name_w;
  static const int data_w;

 public:
  BasicTimer() = default;
  BasicTimer(int id, const std::string& key) : id_(id), key_(key) {}

  void SetId(int id) { id_ = id; }
  void SetKey(const std::string& key) { key_ = key; }
  void Start(const std::string& timer_key) {
    TimerInfo& timer_info = timer_infos_[timer_key];
    timer_info.timer_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  void Stop(const std::string& timer_key) {
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

  void SetCustomInfo(const std::string& key, const std::string& value) {
    if (custom_infos_.find(key) != custom_infos_.end()) {
      LOG(FATAL) << "Error: Custom Info for key [" << key
                 << "] can't be overwritten";
    }
    custom_infos_[key] = value;
  }

  std::string GetCustomInfo(const std::string& key) const {
    auto iter = custom_infos_.find(key);
    if (iter == custom_infos_.end()) {
      LOG(FATAL) << "Error: Custom Info for key [" << key << "] can't be found";
    }
    return iter->second;
  }

  const TimerInfo& GetTimerInfo(const std::string& key) const {
    auto iter = timer_infos_.find(key);
    if (iter == timer_infos_.end()) {
      LOG(FATAL) << "Error: Timer Info for key [" << key << "] can't be found";
    }
    return iter->second;
  }

  void Log(TimerInfo* timer_info, uint32_t timespan) {
    CHECK(timer_info);
    timer_info->total_ += timespan;
    timer_info->max_ = std::max(timer_info->max_, timespan);
    timer_info->min_ = std::min(timer_info->min_, timespan);
    timer_info->count_++;
  }

  static std::string basic_repr_header() {
    STL::stringstream ss;
    // clang-format off
    ss << "op"        << "\t"
       << "kernel"    << "\t"
       << "k_average" << "\t"
       << "k_min"     << "\t"
       << "k_max"     << "\t"
       << "i_average" << "\t"
       << "i_min"     << "\t"
       << "i_max"     << "\t"
       << "count"     << "\t"
       << "op_info";
    // clang-format on
    return ss.str();
  }

  std::string basic_repr() const {
    auto& kernel_timer_info = GetTimerInfo("kernel");
    auto& inst_timer_info = GetTimerInfo("instruction");
    STL::stringstream ss;
    // clang-format off
    ss << GetCustomInfo("op_type") << "\t"
       << key()                    << "\t"
       << kernel_timer_info.ave()  << "\t"
       << kernel_timer_info.min()  << "\t"
       << kernel_timer_info.max()  << "\t"
       << inst_timer_info.ave()    << "\t"
       << inst_timer_info.min()    << "\t"
       << inst_timer_info.max()    << "\t"
       << inst_timer_info.count()  << "\t"
       << GetCustomInfo("op_info");
    // clang-format on
    return ss.str();
  }

  const std::string& key() const { return key_; }

  int id() const {
    CHECK_GE(id_, 0) << "id is not inited";
    return id_;
  }

  // BasicRecord(const BasicRecord &) = delete;
  void operator=(const BasicTimer&) = delete;
};

/*
 * A basic profiler, with each record logs the total latency.
 */
template <typename TimerT>
class BasicProfiler {
 public:
  explicit BasicProfiler(const std::string& name) : name_(name) {}
  using record_t = TimerT;

  static BasicProfiler& Global() {
    static std::unique_ptr<BasicProfiler> x(new BasicProfiler("[global]"));
    return *x;
  }

  record_t& NewRcd(const std::string& key) {
    records_.emplace_back();
    records_.back().SetId(records_.size() - 1);
    records_.back().SetKey(key);
    return records_.back();
  }

  const record_t& record(int id) {
    CHECK_LT(id, records_.size());
    CHECK_GE(id, 0);
    return records_[id];
  }

  record_t* mutable_record(int id) {
    CHECK_GE(id, 0);
    CHECK_LT(static_cast<size_t>(id), records_.size());
    return &records_[id];
  }

  std::string basic_repr() const {
    STL::stringstream ss;
    for (const auto& rcd : records_) {
      ss << rcd.basic_repr() << "\n";
    }
    return ss.str();
  }

  std::string summary_repr_header() const {
    STL::stringstream ss;
    // clang-format off
    ss << "op"         << "\t"
       << "average"    << "\t"
       << "min"        << "\t"
       << "max"        << "\t"
       << "op_time"    << "\t"
       << "total_time" << "\t"
       << "precent"    << "\t"
       << "count";
    // clang-format on
    return ss.str();
  }

  std::string summary_repr() const {
    std::map<std::string, TimerInfo> op_summary;
    uint64_t total{0};

    for (const auto& rcd : records_) {
      // We use kernel run time here
      auto kernel_timer = rcd.GetTimerInfo("kernel");
      auto op_type = rcd.GetCustomInfo("op_type");
      auto& op_timer = op_summary[op_type];

      total += kernel_timer.total_;
      op_timer.total_ += kernel_timer.total_;
      op_timer.max_ = std::max(kernel_timer.max_, op_timer.max_);
      op_timer.min_ = std::min(kernel_timer.min_, op_timer.min_);
      op_timer.count_ += kernel_timer.count_;
    }

    STL::stringstream ss;
    for (auto& iter : op_summary) {
      auto& op_timer = iter.second;
      // clang-format off
      ss << iter.first                             << "\t"
         << op_timer.ave()                         << "\t"
         << op_timer.min()                         << "\t"
         << op_timer.max()                         << "\t"
         << op_timer.total()                       << "\t"
         << total                                  << "\t"
         << (op_timer.total() * 1. / total * 100)  << "%\t"
         << op_timer.count()                       << "\t"
         << "\n";
      // clang-format on
    }
    return ss.str();
  }

  ~BasicProfiler();

 private:
  std::string name_;
  std::vector<record_t> records_;
};

struct ProfileBlock {
  explicit ProfileBlock(int id, const std::string& key) : id_(id), key_(key) {
    BasicProfiler<BasicTimer>::Global().mutable_record(id_)->Start(key_);
  }

  void Record() {
    if (has_recorded_) {
      LOG(FATAL) << "You can only call Record() once";
    }
    BasicProfiler<BasicTimer>::Global().mutable_record(id_)->Stop(key_);
    has_recorded_ = true;
  }

  ~ProfileBlock() {
    if (!has_recorded_) {
      BasicProfiler<BasicTimer>::Global().mutable_record(id_)->Stop(key_);
    }
  }

 private:
  int id_{};
  bool has_recorded_{false};
  std::string key_{};
};

#define LITE_PROFILE_ONE(key__)                            \
  static int key__##__profiler_id =                        \
      ::paddle::lite::profile::BasicProfiler<              \
          ::paddle::lite::profile::BasicTimer>::Global()   \
          .NewRcd(#key__)                                  \
          .id();                                           \
  ::paddle::lite::profile::ProfileBlock key__##profiler__( \
      key__##__profiler_id, #key__);

}  // namespace profile
}  // namespace lite
}  // namespace paddle
