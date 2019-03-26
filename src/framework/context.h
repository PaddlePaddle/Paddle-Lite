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

#if _OPENMP
#include <omp.h>
#endif

#define MOBILE_MAX_CPU_NUM 8

namespace paddle_mobile {
namespace framework {

struct CPUContext {
 private:
  CPUContext() : num_cpus(4), num_threads(1) {
    // TODO(hjchen2)
    for (int i = 0; i < num_cpus; ++i) {
      cpu_frequencies[i] = 2400;      // 2400 MHz
      max_cpu_frequencies[i] = 2400;  // 2400 MHz
    }
    //    L1_cache = 32000;    // 32K
    L1_cache = 32 * 1024;
    L2_cache = 2000000;  // 2M
                         //    L2_cache = 512000;
  }

 public:
  void set_num_threads(int threads) {
#if _ONENMP
    omp_set_num_threads(threads);
    if (threads <= omp_get_max_threads()) {
      num_threads = threads;
    } else {
      num_threads = omp_get_max_threads();
    }
#endif
    num_threads = (num_threads > 1) ? num_threads : 1;
  }

  virtual ~CPUContext() {}

 public:
  static CPUContext* Context() {
    static CPUContext* ctx = new CPUContext;
    return ctx;
  }

  int num_cpus;
  int num_threads;
  int cpu_frequencies[MOBILE_MAX_CPU_NUM];
  int max_cpu_frequencies[MOBILE_MAX_CPU_NUM];

  int L1_cache;
  int L2_cache;
};

inline void set_global_num_threads(int threads) {
  // CPUContext::Context()->set_num_threads(threads);
  CPUContext::Context()->num_threads = threads;
}

inline int get_global_num_threads() {
  return CPUContext::Context()->num_threads;
}

}  // namespace framework
}  // namespace paddle_mobile
