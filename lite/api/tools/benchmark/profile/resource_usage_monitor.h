// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_API_TOOLS_PROFILING_RESOURCE_USAGE_MONITOR_H_
#define LITE_API_TOOLS_PROFILING_RESOURCE_USAGE_MONITOR_H_

#include <unistd.h>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include "cpu_usage_info.h"
#include "memory_info.h"

namespace paddle {
namespace lite_api {
namespace profile {

// This class could help to tell the peak resource footprint of a running
// program.
// It achieves this by spawning a thread to check the resource usage
// periodically
// at a pre-defined frequency.
class ResourceUsageMonitor {
 public:
  // A helper class that does resource usage sampling. This allows injecting an
  // external dependency for the sake of testing or providing platform-specific
  // implementations.
  class Sampler {
   public:
    ~Sampler() {}

    bool IsSupported() { return MemoryUsage::IsSupported(); }

    MemoryUsage GetMemoryUsage() {
      return paddle::lite_api::profile::GetMemoryUsage();
    }

    float GetCpuUsageRatio(int pid) {
      return paddle::lite_api::profile::GetCpuUsageRatio(pid);
    }

    void SleepFor(const int duration) {
      std::this_thread::sleep_for(std::chrono::milliseconds(duration));
    }
  };

  static constexpr float kInvalidMemUsageKB = -1.0f;

  explicit ResourceUsageMonitor(int sampling_interval_ms = 10)
      : ResourceUsageMonitor(sampling_interval_ms,
                             std::unique_ptr<Sampler>(new Sampler())) {}
  ResourceUsageMonitor(int sampling_interval_ms,
                       std::unique_ptr<Sampler> sampler);
  ~ResourceUsageMonitor() { StopInternal(); }

  void Start();
  void Stop();

  // For simplicity, we will return kInvalidMemUsageKB for the either following
  // conditions:
  // 1. getting memory usage isn't supported on the platform.
  // 2. the memory usage is being monitored (i.e. we've created the
  // 'check_memory_thd_'.
  float GetPeakMemUsageInKB() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return kInvalidMemUsageKB;
    }
    return peak_max_rss_kb_;
  }

  ResourceUsageMonitor(ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor(ResourceUsageMonitor&&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&&) = delete;

 private:
  void StopInternal();

  std::unique_ptr<Sampler> sampler_ = nullptr;
  bool is_supported_ = false;
  bool stop_signal_ = false;
  const int sampling_interval_;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
  int64_t peak_max_rss_kb_ = kInvalidMemUsageKB;
};

}  // namespace paddle
}  // namespace lite_api
}  // namespace profile

#endif  // LITE_API_TOOLS_PROFILING_RESOURCE_USAGE_MONITOR_H_
