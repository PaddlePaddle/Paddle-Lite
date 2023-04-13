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
#include "arm_dnn_library/core/types.h"

#if ARM_DNN_LIBRARY_OS_IOS
const int DEFAULT_L1_CACHE_SIZE = 64 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#elif ARM_DNN_LIBRARY_OS_MAC && ARM_DNN_LIBRARY_ARCH_ARM64  // M1
const int DEFAULT_L1_CACHE_SIZE = 128 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 4096 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#else
const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 512 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#endif  // ARM_DNN_LIBRARY_OS_IOS

namespace armdnnlibrary {

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
  size_t count() { return cpu_attrs_.size(); }
  const CPUAttr& at(int index);
  PowerMode power_mode() { return power_mode_; }
  size_t thread_num() { return active_idxs_.size(); }
  CPUArch arch() { return cpu_attrs_[active_idxs_[0]].arch; }
  int32_t cluster_id() { return cpu_attrs_[active_idxs_[0]].cluster_id; }
  size_t l1_cache_size() { return cpu_attrs_[active_idxs_[0]].l1_cache_size; }
  size_t l2_cache_size() { return cpu_attrs_[active_idxs_[0]].l2_cache_size; }
  size_t l3_cache_size() { return cpu_attrs_[active_idxs_[0]].l3_cache_size; }
  bool support_arm_fp16() {
    return cpu_attrs_[active_idxs_[0]].support_arm_fp16;
  }
  bool support_arm_bf16() {
    return cpu_attrs_[active_idxs_[0]].support_arm_bf16;
  }
  bool support_arm_dotprod() {
    return cpu_attrs_[active_idxs_[0]].support_arm_dotprod;
  }
  bool support_arm_sve2() {
    return cpu_attrs_[active_idxs_[0]].support_arm_sve2;
  }
  bool support_arm_sve2_i8mm() {
    return cpu_attrs_[active_idxs_[0]].support_arm_sve2_i8mm;
  }
  bool support_arm_sve2_f32mm() {
    return cpu_attrs_[active_idxs_[0]].support_arm_sve2_f32mm;
  }
  static CPUInfo& Singleton();
  bool Initialize();
  bool SetRunMode(PowerMode power_mode, size_t thread_num);
  void DumpAllInfo();

 private:
  std::vector<CPUAttr> cpu_attrs_;
  static ARM_DNN_LIBRARY_THREAD_LOCAL PowerMode power_mode_;
  static ARM_DNN_LIBRARY_THREAD_LOCAL std::vector<int> active_idxs_;
};

}  // namespace armdnnlibrary
