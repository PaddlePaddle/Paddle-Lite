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
#include "utilities/logging.h"

namespace adnn {

class Device {
 public:
  explicit Device(const Callback* callback);
  ~Device();
  Status SetParam(ParamKey key, ParamValue value);
  Status GetParam(ParamKey key, ParamValue* value);
  const Callback* GetCallback() {
    ADNN_CHECK(callback_);
    return callback_;
  }
  void* GetDevice() { return device_; }
  // Helper functions for querying the parameters.
  int32_t GetMaxThreadNum();
  bool GetSupportArmFP16();
  bool GetSupportArmBF16();
  bool GetSupportArmDotProd();
  bool GetSupportArmSVE2();

 private:
  const Callback* callback_{nullptr};
  void* device_{nullptr};
  std::unordered_map<int, ParamValue> params_;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

}  // namespace adnn
