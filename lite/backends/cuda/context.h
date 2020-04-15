// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>
#include "lite/backends/cuda/blas.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/target_wrapper.h"
#include "lite/core/device_info.h"

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;

using CUDAContext = Context<TargetType::kCUDA>;

// Only works with CUDA kernels.
template <>
class Context<TargetType::kCUDA> {
 public:
  typename Env<TargetType::kCUDA>::Devs& devs =
      Env<TargetType::kCUDA>::Global();
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {
    if (devs.size() > 0) {
      cublas_fp32_ = std::make_shared<lite::cuda::Blas<float>>();
    } else {
      LOG(INFO) << "No cuda device(s) found, CUDAContext init failed.";
    }
  }
  void Init(int dev_id, int exec_stream_id = 0, int io_stream_id = 0) {
    CHECK_GT(devs.size(), 0UL)
        << "Env is not initialized or current target is not exit!";
    if (dev_id >= static_cast<int>(devs.size())) {
      LOG(WARNING) << "device index exceeds the number of devices, set to "
                      "default device(0)!";
      device_id_ = 0;
    } else {
      device_id_ = dev_id;
    }
    if (io_stream_id >= devs[dev_id].max_stream()) {
      LOG(WARNING) << "data stream index exceeds the maximum stream number, "
                      "set to default stream(0)!";
      io_stream_id = 0;
    }
    if (exec_stream_id >= devs[dev_id].max_stream()) {
      LOG(WARNING) << "exec stream index exceeds the maximum stream number, "
                      "set to default stream(0)!";
      exec_stream_id = 0;
    }

    exec_stream_ = devs[dev_id].exec_streams()[exec_stream_id];
    io_stream_ = devs[dev_id].io_streams()[io_stream_id];

    exec_stream_id_ = exec_stream_id;
    io_stream_id_ = io_stream_id;
    need_sync_ = false;
  }
  void CopySharedTo(CUDAContext* ctx) {
    CHECK(ctx);
    CHECK(cublas_fp32_) << "cublas_fp32 should be set first";
    ctx->cublas_fp32_ = cublas_fp32_;
  }

  const cudaStream_t& exec_stream() const { return exec_stream_; }
  void SetExecStream(cudaStream_t stream) { exec_stream_ = stream; }

  const cudaStream_t& io_stream() const { return io_stream_; }
  void SetIoStream(cudaStream_t stream) { io_stream_ = stream; }

  std::shared_ptr<cuda::Blas<float>> cublas_fp32() { return cublas_fp32_; }
  void SetCuBlasFP32(std::shared_ptr<cuda::Blas<float>> cublas_fp32) {
    cublas_fp32_ = cublas_fp32;
  }

  const std::vector<cudaEvent_t>& input_events() { return input_events_; }
  void SetInputEvents(const std::vector<cudaEvent_t>& input_events) {
    input_events_.clear();
    input_events_.assign(input_events.begin(), input_events.end());
  }

  const std::vector<cudaEvent_t>& output_events() { return output_events_; }
  void SetOutputEvents(const std::vector<cudaEvent_t>& output_events) {
    output_events_.clear();
    output_events_.assign(output_events.begin(), output_events.end());
  }

  std::vector<cudaStream_t> all_exec_streams() {
    int dev_id = TargetWrapper<TargetType::kCUDA>::GetCurDevice();
    return devs[dev_id].exec_streams();
  }

  void SetSyncStreams(const std::vector<int>& nums) {
    sync_streams_.clear();
    std::vector<cudaStream_t> exec_streams = all_exec_streams();
    for (size_t i = 0; i < nums.size(); ++i) {
      CHECK(nums[i] >= 0 && nums[i] < static_cast<int>(exec_streams.size()))
          << "streams id is not valid";
      sync_streams_.push_back(exec_streams[nums[i]]);
    }
    InitSyncEvents(nums.size());
  }

  void InitSyncEvents(const int num) {
    sync_events_.clear();
    for (int i = 0; i < num; ++i) {
      cudaEvent_t eve;
      TargetWrapperCuda::CreateEventWithFlags(&eve);
      sync_events_.push_back(eve);
    }
  }

  void SetNeedSync(bool sync) { need_sync_ = sync; }
  bool need_sync() const { return need_sync_; }

  void Sync() {
    CHECK_EQ(sync_streams_.size(), sync_events_.size());
    for (size_t i = 0; i < sync_events_.size(); ++i) {
      TargetWrapperCuda::RecordEvent(sync_events_[i], sync_streams_[i]);
      TargetWrapperCuda::StreamSync(exec_stream_, sync_events_[i]);
    }
  }

  std::string name() const { return "CUDAContext"; }

  CUDAContext& operator=(const CUDAContext& context) {
    this->Init(
        context.device_id_, context.exec_stream_id_, context.io_stream_id_);
    cublas_fp32_ = const_cast<CUDAContext&>(context).cublas_fp32();
    return *this;
  }

 private:
  int device_id_;
  // overall information
  int exec_stream_id_;
  int io_stream_id_;
  cudaStream_t exec_stream_;
  cudaStream_t io_stream_;

  // not thread-safe, should allocate for each thread.
  std::shared_ptr<cuda::Blas<float>> cublas_fp32_;

  // kernel information
  std::vector<cudaEvent_t> input_events_;
  std::vector<cudaEvent_t> output_events_;
  // multi stream sync.
  std::vector<cudaStream_t> sync_streams_;
  std::vector<cudaEvent_t> sync_events_;
  bool need_sync_;
};

}  // namespace lite
}  // namespace paddle
