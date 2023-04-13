// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>
#include "arm_dnn_library/core/types.h"
#include "runtime/device.h"

namespace armdnnlibrary {

class Context {
 public:
  explicit Context(Device* device);
  ~Context();
  Status SetParam(ParamKey key, ParamValue value, bool force = false);
  Status GetParam(ParamKey key, ParamValue* value);
  void* MemoryAlloc(size_t size);
  void MemoryFree(void* ptr);
  void* MemoryAlignedAlloc(size_t size, size_t alignment = 64);
  void MemoryAlignedFree(void* ptr);
  Device* device() { return device_; }
  void* context() { return context_; }
  // Helper functions for querying the parameters.
  int32_t work_thread_num();
  bool enable_arm_fp16();
  bool enable_arm_bf16();
  bool enable_arm_dotprod();
  bool enable_arm_sve2();
  bool enable_arm_sve2_i8mm();
  bool enable_arm_sve2_f32mm();
  void* workspace_data();
  size_t workspace_size();
  void* SetWorkspaceSize(size_t size);

 private:
  Device* device_{nullptr};
  void* context_{nullptr};
  std::unordered_map<int, ParamValue> params_;
  static ARM_DNN_LIBRARY_THREAD_LOCAL void* workspace_data_;
  static ARM_DNN_LIBRARY_THREAD_LOCAL size_t workspace_size_;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

}  // namespace armdnnlibrary
