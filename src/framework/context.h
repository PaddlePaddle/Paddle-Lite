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

#include <vector>
#include "framework/tensor.h"

namespace paddle_mobile {
namespace framework {

struct CPUContext {
 private:
  CPUContext();
  virtual ~CPUContext() {}

 public:
  static CPUContext* Context() {
    static CPUContext* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = new CPUContext();
    }
    return ctx;
  }

  void set_thread_num(int thread_num,
                      PowerMode power_mode = PERFORMANCE_PRIORITY);
  int get_thread_num();
  PowerMode get_power_mode() const { return _power_mode; }
  int get_cache_size(int level);
  ARMArch get_arch() const { return _arch; }
  int get_l1_cache_size() { return get_cache_size(1); }
  int get_l2_cache_size() { return get_cache_size(2); }
  int get_l3_cache_size() { return get_cache_size(3); }
  void* get_work_space(int size_in_byte);

  int _cpu_num;
  ARMArch _arch;
  PowerMode _power_mode;
  std::vector<int> _big_core_ids;
  std::vector<int> _little_core_ids;
  std::vector<int> _l1_cache_sizes;
  std::vector<int> _l2_cache_sizes;
  std::vector<int> _l3_cache_sizes;
  Tensor _workspace;
};

}  // namespace framework
}  // namespace paddle_mobile
