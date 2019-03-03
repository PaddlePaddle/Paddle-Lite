/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#define MOBILE_MAX_CPU_NUM 8

namespace paddle_mobile {
namespace operators {
namespace math {

struct CPUInfo {
 private:
  CPUInfo() {
    // TODO(hjchen2)
    num_cpus = 4;
    for (int i = 0; i < num_cpus; ++i) {
      cpu_frequency[i] = 2400;      // 2400 MHz
      max_cpu_frequency[i] = 2400;  // 2400 MHz
    }
    //    L1_cache = 32000;    // 32K
    L1_cache = 32 * 1024;
    L2_cache = 2000000;  // 2M
                         //    L2_cache = 512000;
  }
  virtual ~CPUInfo() {}

 public:
  static CPUInfo* Info() {
    static CPUInfo* ctx = new CPUInfo;
    return ctx;
  }

  int num_cpus;
  int cpu_frequency[MOBILE_MAX_CPU_NUM];
  int max_cpu_frequency[MOBILE_MAX_CPU_NUM];

  int L1_cache;
  int L2_cache;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
