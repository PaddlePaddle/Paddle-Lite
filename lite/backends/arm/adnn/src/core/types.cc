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

#include "adnn/core/types.h"
#include <vector>

namespace adnn {

#define ADNN_TYPE_TO_STRING(type) \
  case type:                      \
    name = #type;                 \
    break;

ADNN_DLL_EXPORT std::string status_to_string(Status type) {
  std::string name;
  switch (type) {
    ADNN_TYPE_TO_STRING(SUCCESS);
    ADNN_TYPE_TO_STRING(OUT_OF_MEMORY);
    ADNN_TYPE_TO_STRING(INVALID_PARAMETER);
    ADNN_TYPE_TO_STRING(FEATURE_NOT_SUPPORTED);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

ADNN_DLL_EXPORT std::string power_mode_to_string(PowerMode type) {
  std::string name;
  switch (type) {
    ADNN_TYPE_TO_STRING(HIGH);
    ADNN_TYPE_TO_STRING(LOW);
    ADNN_TYPE_TO_STRING(FULL);
    ADNN_TYPE_TO_STRING(NO_BIND);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

ADNN_DLL_EXPORT std::string cpu_arch_to_string(CPUArch type) {
  std::string name;
  switch (type) {
    ADNN_TYPE_TO_STRING(CORTEX_A35);
    ADNN_TYPE_TO_STRING(CORTEX_A53);
    ADNN_TYPE_TO_STRING(CORTEX_A55);
    ADNN_TYPE_TO_STRING(CORTEX_A57);
    ADNN_TYPE_TO_STRING(CORTEX_A72);
    ADNN_TYPE_TO_STRING(CORTEX_A73);
    ADNN_TYPE_TO_STRING(CORTEX_A75);
    ADNN_TYPE_TO_STRING(CORTEX_A76);
    ADNN_TYPE_TO_STRING(CORTEX_A77);
    ADNN_TYPE_TO_STRING(CORTEX_A78);
    ADNN_TYPE_TO_STRING(CORTEX_X1);
    ADNN_TYPE_TO_STRING(CORTEX_X2);
    ADNN_TYPE_TO_STRING(CORTEX_A510);
    ADNN_TYPE_TO_STRING(CORTEX_A710);
    ADNN_TYPE_TO_STRING(KRYO_485_GOLD_PRIME);
    ADNN_TYPE_TO_STRING(KRYO_485_GOLD);
    ADNN_TYPE_TO_STRING(KRYO_485_SILVER);
    ADNN_TYPE_TO_STRING(APPLE);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

ADNN_DLL_EXPORT std::string param_key_to_string(ParamKey type) {
  std::string name;
  switch (type) {
    ADNN_TYPE_TO_STRING(DEVICE_MAX_THREAD_NUM);
    ADNN_TYPE_TO_STRING(DEVICE_POWER_MODE);
    ADNN_TYPE_TO_STRING(DEVICE_ARCH);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_FP16);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_BF16);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_DOTPROD);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_SVE2);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_SVE2_I8MM);
    ADNN_TYPE_TO_STRING(DEVICE_SUPPORT_ARM_SVE2_F32MM);
    ADNN_TYPE_TO_STRING(CONTEXT_WORK_THREAD_NUM);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_FP16);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_BF16);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_DOTPROD);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_SVE2);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_SVE2_I8MM);
    ADNN_TYPE_TO_STRING(CONTEXT_ENABLE_ARM_SVE2_F32MM);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

}  // namespace adnn
