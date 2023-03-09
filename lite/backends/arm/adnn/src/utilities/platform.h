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

namespace adnn {

// Get the CPU info.
int get_cpu_num();
std::string get_cpu_name();
size_t get_mem_size();
void get_cpu_arch(std::vector<CPUArch>* cpu_archs, const int cpu_num);
int get_min_freq_khz(int cpu_id);
int get_max_freq_khz(int cpu_id);
void sort_cpuid_by_max_freq(const std::vector<int>& max_freqs,
                            std::vector<int>* cpu_ids,
                            std::vector<int>* cluster_ids);
void get_cpu_cache_size(int cpu_id,
                        int* l1_cache_size,
                        int* l2_cache_size,
                        int* l3_cache_size);
bool check_cpu_online(const std::vector<int>& cpu_ids);
bool set_sched_affinity(const std::vector<int>& cpu_ids);
bool bind_threads(const std::vector<int> cpu_ids);

// Check the CPU features.
bool support_arm_sve2();
bool support_arm_sve2_i8mm();
bool support_arm_sve2_f32mm();

}  // namespace adnn
