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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// The environment variables for selecting Huawei Ascend npu device ids, use
// "HUAWEI_" as prefix. Specify the device ids of Huawei Ascend npu for model
// inference, an example is shown as below:
// HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS="0,1,2,3"
// HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS="0"
#define HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS \
  "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS"

namespace nnadapter {

std::string GetStringFromEnv(const std::string& str,
                             const std::string& def = "");
bool GetBoolFromEnv(const std::string& str, bool def = false);
int GetIntFromEnv(const std::string& str, int def = 0);
double GetDoubleFromEnv(const std::string& str, double def = 0.0);
uint64_t GetUInt64FromEnv(const std::string& str, uint64_t def = 0ul);

}  // namespace nnadapter
