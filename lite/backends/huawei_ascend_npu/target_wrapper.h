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
#include <mutex>  //NOLINT
#include <utility>
#include <vector>
#include "lite/backends/huawei_ascend_npu/utils.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperHuaweiAscendNPU = TargetWrapper<TARGET(kHuaweiAscendNPU)>;

template <>
class TargetWrapper<TARGET(kHuaweiAscendNPU)> {
 public:
  using context_t = aclrtContext;
  using stream_t = aclrtStream;
  using event_t = aclrtEvent;

  static size_t num_devices();
  static size_t maximum_stream() { return 1024; }

  static void CreateDevice(int device_id);
  static void DestroyDevice(int device_id);
  static int GetCurDevice() { return device_id_; }

  static void CreateContext(aclrtContext* context) {}
  static void DestroyContext(const aclrtContext& context) {}

  static void CreateStream(aclrtStream* stream) {}
  static void DestroyStream(const aclrtStream& stream) {}

  static void CreateEvent(aclrtEvent* event) {}
  static void DestroyEvent(const aclrtEvent& event) {}

  static void* MallocHost(size_t size) { return nullptr; }
  static void FreeHost(void* ptr) {}

  static void* MallocDevice(size_t size) { return nullptr; }
  static void FreeDevice(void* ptr) {}

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir) {}

  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream) {}

  static void MemsetSync(void* devPtr, int value, size_t count) {}

  static void MemsetAsync(void* devPtr,
                          int value,
                          size_t count,
                          const stream_t& stream) {}

 private:
  static void InitOnce();
  static void DestroyOnce();

 private:
  static bool runtime_inited_;
  static std::vector<int> device_list_;
  static std::mutex device_mutex_;

  static thread_local int device_id_;
};

}  // namespace lite
}  // namespace paddle
