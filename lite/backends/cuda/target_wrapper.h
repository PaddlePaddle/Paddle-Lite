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
#include <cuda.h>
#include <cuda_runtime.h>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperCuda = TargetWrapper<TARGET(kCUDA)>;

template <>
class TargetWrapper<TARGET(kCUDA)> {
 public:
  using stream_t = cudaStream_t;
  using event_t = cudaEvent_t;

  static size_t num_devices();
  static size_t maximum_stream() { return 0; }

  static int GetComputeCapability() {
    int dev_id = GetCurDevice();
    int major, minor;
    CUDA_CALL(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, dev_id));
    CUDA_CALL(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, dev_id));
    return major * 10 + minor;
  }

  static size_t GetCurDevice() {
    int dev_id;
    cudaGetDevice(&dev_id);
    return dev_id;
  }
  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) { cudaEventCreate(event); }
  static void CreateEventWithFlags(
      event_t* event, unsigned int flags = cudaEventDisableTiming) {
    cudaEventCreateWithFlags(event, flags);
  }
  static void DestroyEvent(const event_t& event) { cudaEventDestroy(event); }

  static void RecordEvent(const event_t& event) {}
  static void RecordEvent(const event_t& event, const stream_t& stream) {
    cudaEventRecord(event, stream);
  }
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {
    cudaStreamSynchronize(stream);
  }
  static void StreamSync(const stream_t& stream, const event_t& event) {
    cudaStreamWaitEvent(stream, event, 0);
  }
  static void DeviceSync() { cudaDeviceSynchronize(); }

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream);

  static void MemsetSync(void* devPtr, int value, size_t count);

  static void MemsetAsync(void* devPtr,
                          int value,
                          size_t count,
                          const stream_t& stream);
};
}  // namespace lite
}  // namespace paddle
