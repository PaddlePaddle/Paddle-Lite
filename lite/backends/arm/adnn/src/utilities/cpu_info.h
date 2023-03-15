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

#include <string>
#include <vector>
#include "adnn/core/types.h"

#if ADNN_OS_IOS
const int DEFAULT_L1_CACHE_SIZE = 64 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#elif ADNN_OS_MAC && ADNN_ARCH_ARM64  // M1
const int DEFAULT_L1_CACHE_SIZE = 128 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 4096 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#else
const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 512 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#endif  // ADNN_OS_IOS

namespace adnn {

#define CPU_ATTR_ARCH 0
#define CPU_ATTR_CLUSTER_ID 1
#define CPU_ATTR_L1_CACHE_SIZE 2
#define CPU_ATTR_L2_CACHE_SIZE 3
#define CPU_ATTR_L3_CACHE_SIZE 4
#define CPU_ATTR_SUPPORT_ARM_FP16 5
#define CPU_ATTR_SUPPORT_ARM_BF16 6
#define CPU_ATTR_SUPPORT_ARM_DOTPROD 7
#define CPU_ATTR_SUPPORT_ARM_SVE2 8
#define CPU_ATTR_SUPPORT_ARM_SVE2_I8MM 9
#define CPU_ATTR_SUPPORT_ARM_SVE2_F32MM 10

typedef struct {
  // [0] CPU arch
  CPUArch arch{CPUArch::UNKOWN};
  // [1] CPU cluster
  // 0 = LITTLE, 1 = Middle, 2 = big
  int32_t cluster_id{0};
  // [2] L1 cache size
  size_t l1_cache_size{DEFAULT_L1_CACHE_SIZE};
  // [3] L2 cache size
  size_t l2_cache_size{DEFAULT_L2_CACHE_SIZE};
  // [4] L3 cache size
  size_t l3_cache_size{DEFAULT_L3_CACHE_SIZE};
  // [5] Does support arm fp16 intruction
  bool support_arm_fp16{false};
  // [6] Does support arm bf16 intruction
  bool support_arm_bf16{false};
  // [7] Does support arm dotprod intruction
  bool support_arm_dotprod{false};
  // [8] Does support arm sve2 intruction
  bool support_arm_sve2{false};
  // [9] Does support arm sve2+i8mm intruction
  bool support_arm_sve2_i8mm{false};
  // [10] Does support arm sve2+f32mm intruction
  bool support_arm_sve2_f32mm{false};
} CPUAttr;

class CPUInfo {
 public:
  CPUInfo();
  ~CPUInfo();
  CPUInfo(const CPUInfo&) = delete;
  CPUInfo& operator=(const CPUInfo&) = delete;
  static size_t count();
  static const CPUAttr& at(int index);
  static void dump();
  static PowerMode power_mode();
  static bool SetPowerMode(PowerMode power_mode, size_t thread_num);

 private:
  static CPUInfo& Singleton();
  std::vector<CPUAttr> cpu_attrs_;
  PowerMode power_mode_;
  std::vector<int> active_ids_;
};

}  // namespace adnn
