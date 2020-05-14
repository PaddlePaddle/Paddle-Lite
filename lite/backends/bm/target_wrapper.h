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
#include <map>
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperBM = TargetWrapper<TARGET(kBM)>;

template <>
class TargetWrapper<TARGET(kBM)> {
 public:
  using stream_t = int;
  using event_t = int;

  static size_t num_devices();
  static size_t maximum_stream() { return 0; }

  static void SetDevice(int id);
  static int GetDevice();
  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void* GetHandle();

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

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
  static int device_id_;
  static std::map<int, void*> bm_hds_;
};
}  // namespace lite
}  // namespace paddle
