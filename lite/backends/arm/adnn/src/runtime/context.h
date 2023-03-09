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
#include "adnn/core/types.h"
#include "runtime/device.h"

namespace adnn {

class Context {
 public:
  explicit Context(Device* device);
  ~Context();
  Status SetParam(ParamKey key, ParamValue value);
  Status GetParam(ParamKey key, ParamValue* value);
  void* MemoryAlloc(size_t size);
  void MemoryFree(void* ptr);
  void* MemoryAlignedAlloc(size_t alignment, size_t size);
  void MemoryAlignedFree(void* ptr);
  Device* GetDevice() { return device_; }
  void* GetContext() { return context_; }
  // Helper functions for querying the parameters.
  int32_t GetWorkThreadNum();
  bool GetEnableArmFP16();
  bool GetEnableArmBF16();
  bool GetEnableArmDotProd();
  bool GetEnableArmSVE2();

 private:
  Device* device_{nullptr};
  void* context_{nullptr};
  std::unordered_map<int, ParamValue> params_;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

}  // namespace adnn
