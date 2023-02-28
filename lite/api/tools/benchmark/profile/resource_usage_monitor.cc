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

#include "lite/api/tools/benchmark/profile/resource_usage_monitor.h"
#include <iomanip>
#include <iostream>
#include <utility>

namespace paddle {
namespace lite_api {
namespace profile {

constexpr float ResourceUsageMonitor::kInvalidMemUsageKB;

ResourceUsageMonitor::ResourceUsageMonitor(int sampling_interval_ms,
                                           std::unique_ptr<Sampler> sampler)
    : sampler_(std::move(sampler)),
      is_supported_(false),
      sampling_interval_(sampling_interval_ms) {
  is_supported_ = (sampler_ != nullptr && sampler_->IsSupported());
  if (!is_supported_) {
    std::cout << "Getting memory usage isn't supported on this platform!"
              << std::endl;
    return;
  }
}

void ResourceUsageMonitor::Start() {
  if (!is_supported_) return;
  if (check_memory_thd_ != nullptr) {
    std::cout << "Memory monitoring has already started!" << std::endl;
    return;
  }
  std::cout << "start monitoring memory!" << std::endl;
  stop_signal_ = false;
  check_memory_thd_.reset(new std::thread(([this]() {
    // Note we retrieve the memory usage at the very beginning of the thread.
    while (true) {
      const auto mem_info = sampler_->GetMemoryUsage();
      if (mem_info.max_rss_kb > peak_max_rss_kb_) {
        peak_max_rss_kb_ = mem_info.max_rss_kb;
      }
      std::cout << std::fixed << "cpu usage ratio: " << std::setprecision(1)
                << sampler_->GetCpuUsageRatio(getpid()) * 100 << "%"
                << std::endl;
      if (stop_signal_) break;
      sampler_->SleepFor(sampling_interval_);
    }
  })));
}

void ResourceUsageMonitor::Stop() {
  if (!is_supported_) return;
  if (check_memory_thd_ == nullptr) {
    std::cout << "Memory monitoring hasn't started yet or has stopped!"
              << std::endl;
    return;
  }
  std::cout << "stop monitoring memory!" << std::endl;
  StopInternal();
}

void ResourceUsageMonitor::StopInternal() {
  stop_signal_ = true;
  if (check_memory_thd_ == nullptr) return;
  if (check_memory_thd_ != nullptr) {
    check_memory_thd_->join();
  }
  check_memory_thd_.reset(nullptr);
}

}  // namespace profile
}  // namespace lite_api
}  // namespace paddle
