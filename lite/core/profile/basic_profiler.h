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
#ifndef LITE_WITH_XCODE
#include <gflags/gflags.h>
#endif
namespace paddle {
namespace lite {
namespace profile {

struct TimerInfo {
  uint64_t total_{0};
  uint64_t count_{0};
  uint64_t max_{(std::numeric_limits<uint64_t>::min)()};
  uint64_t min_{(std::numeric_limits<uint64_t>::max)()};
  uint64_t timer_{0};

  double Ave() const { return total_ * 1. / count_; }
  double Max() const { return max_; }
  double Min() const { return min_; }
  uint64_t Total() const { return total_; }
  uint64_t Count() const { return count_; }
};

/* Base class of all the profile records */
template <typename ChildT>
class TimerBase {
 public:
  void Start(const std::string& key) { self()->Start(key); }
  void Stop(const std::string& key) { self()->Stop(key); }
  void Log(TimerInfo* timer_info, uint64_t x) {
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
  int warmup_{0};
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

  int id() const {
    CHECK_GE(id_, 0) << "id is not inited";
    return id_;
  }

  void SetKey(const std::string& key) { key_ = key; }
  const std::string& key() const { return key_; }

  void Start(const std::string& timer_key);
  void Stop(const std::string& timer_key);

  void Log(TimerInfo* timer_info, uint64_t timespan);

  void SetCustomInfo(const std::string& key, const std::string& value);
  std::string GetCustomInfo(const std::string& key) const;

  const TimerInfo& GetTimerInfo(const std::string& key) const;

  static std::string basic_repr_header();
  std::string basic_repr() const;

  // BasicRecord(const BasicRecord &) = delete;
  void operator=(const BasicTimer&) = delete;

  void SetWarmup(int warmup_times);
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
    records_.back().SetWarmup(warmup_);
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

  std::string summary_repr_header() const;
  std::string summary_repr() const;

  void SetWarmup(int warmup_times) {
    CHECK_GE(warmup_times, 0) << "warmup times must >= 0";
    // Instruction and kernel share the common BasicTimer instance, so the
    // warmup count
    // will be decrease twice when instruction execute once
    // TODO(sangoly): fix the ugly code.
    warmup_ = warmup_times * 2;
  }

  ~BasicProfiler();

 private:
  std::string name_;
  std::vector<record_t> records_;
  int warmup_{0};
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
