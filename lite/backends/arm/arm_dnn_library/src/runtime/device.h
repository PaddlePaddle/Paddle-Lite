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
#include "utilities/cpu_info.h"
#include "utilities/logging.h"

namespace armdnnlibrary {

class Device {
 public:
  explicit Device(const Callback* callback);
  ~Device();
  Status SetParam(ParamKey key, ParamValue value);
  Status GetParam(ParamKey key, ParamValue* value);
  const Callback* callback() {
    ARM_DNN_LIBRARY_CHECK(callback_);
    return callback_;
  }
  void* device() { return device_; }
  // Helper functions for querying the parameters.
  int32_t max_thread_num();
  PowerMode power_mode();
  CPUArch arch();
  bool support_arm_fp16();
  bool support_arm_bf16();
  bool support_arm_dotprod();
  bool support_arm_sve2();
  bool support_arm_sve2_i8mm();
  bool support_arm_sve2_f32mm();

 private:
  const Callback* callback_{nullptr};
  void* device_{nullptr};
  std::unordered_map<int, ParamValue> params_;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

}  // namespace armdnnlibrary
