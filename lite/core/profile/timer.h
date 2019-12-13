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
#include <algorithm>
#include <chrono>  // NOLINT
#include <list>
#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/cuda_utils.h"
#endif
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace profile {

template <typename T>
class TimeList {
 public:
  void Clear() { laps_t_.clear(); }
  void Add(T t) { laps_t_.push_back(t); }
  T Last() const { return laps_t_.back(); }
  T Max() const { return *std::max_element(laps_t_.begin(), laps_t_.end()); }
  T Min() const { return *std::min_element(laps_t_.begin(), laps_t_.end()); }
  T Sum() const { return std::accumulate(laps_t_.begin(), laps_t_.end(), 0.0); }
  size_t Size() const { return laps_t_.size(); }
  T Avg() const {
    if (!Size()) {
      return 0;
    }
    return Sum() / Size();
  }
  const std::list<T>& Raw() const { return laps_t_; }

 private:
  std::list<T> laps_t_;
};

class Timer {
 public:
  Timer() = default;
  virtual ~Timer() = default;

  void Reset() { laps_t_.Clear(); }
  void Start() { t_start_ = std::chrono::system_clock::now(); }
  float Stop() {
    t_stop_ = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t_stop_ -
                                                                    t_start_);
    float elapse_ms = 1000.f * static_cast<float>(ts.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den;
    this->laps_t_.Add(elapse_ms);
    return elapse_ms;
  }
  virtual void Start(KernelContext* ctx) { return Start(); }
  virtual float Stop(KernelContext* ctx) { return Stop(); }
  float AvgLapTimeMs() const { return laps_t_.Avg(); }
  const TimeList<float>& LapTimes() const { return laps_t_; }

 protected:
  std::chrono::time_point<std::chrono::system_clock> t_start_, t_stop_;
  TimeList<float> laps_t_;
};

template <TargetType Target>
class DeviceTimer final : public Timer {};

#ifdef LITE_WITH_CUDA
template <>
class DeviceTimer<TargetType::kCUDA> final : public Timer {
 public:
  DeviceTimer() {
    CUDA_CALL(cudaEventCreate(&e_start_));
    CUDA_CALL(cudaEventCreate(&e_stop_));
  }
  ~DeviceTimer() {
    CUDA_CALL(cudaEventDestroy(e_start_));
    CUDA_CALL(cudaEventDestroy(e_stop_));
  }
  void Start(KernelContext* ctx) {
    cudaStream_t stream;
    stream = ctx->As<CUDAContext>().exec_stream();
    CUDA_CALL(cudaEventRecord(e_start_, stream));
  }
  float Stop(KernelContext* ctx) {
    cudaStream_t stream;
    stream = ctx->As<CUDAContext>().exec_stream();
    CUDA_CALL(cudaEventRecord(e_stop_, stream));
    CUDA_CALL(cudaEventSynchronize(e_stop_));
    float elapse_ms = 1.f;
    CUDA_CALL(cudaEventElapsedTime(&elapse_ms, e_start_, e_stop_));
    this->laps_t_.Add(elapse_ms);
    return elapse_ms;
  }

 private:
  cudaEvent_t e_start_, e_stop_;
};
#endif

}  // namespace profile
}  // namespace lite
}  // namespace paddle
