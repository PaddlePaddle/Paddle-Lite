/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "framework/context.h"
#include <iostream>
#include <string>
#include "common/log.h"

#ifdef __APPLE__
#include "TargetConditionals.h"
#ifdef TARGET_OS_IPHONE
// iOS
#elif TARGET_OS_MAC
// Mac OS
#else
// Unsupported platform
#endif
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#else  // Linux or Android
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace paddle_mobile {
namespace framework {

const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;

void fill_cpu_cache_size(std::vector<int> *cpu_cache_sizes, int value,
                         const std::vector<int> cpu_ids = {}) {
  int num = cpu_ids.size();
  if (num > 0) {
    for (int i = 0; i < num; i++) {
      if (cpu_ids.size() > i) {
        int idx = cpu_ids[i];
        if (cpu_cache_sizes->size() > idx) {
          (*cpu_cache_sizes)[idx] = value;
        }
      }
    }
  } else {
    num = cpu_cache_sizes->size();
    for (int i = 0; i < num; i++) {
      if (cpu_cache_sizes->size() > i) {
        (*cpu_cache_sizes)[i] = value;
      }
    }
  }
}

int get_cpu_num() {
#ifdef __APPLE__
  int count = 0;
  size_t len = sizeof(count);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
  if (count < 1) {
    count = 1;
  }
  return count;
#else  // Linux or Android
  // get cpu num from /sys/devices/system/cpu/cpunum/uevent
  int max_cpu_num = 20;
  int count = 0;
  for (int i = 0; i < max_cpu_num; i++) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", i);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
      break;
    }
    count++;
    fclose(fp);
  }
  if (count < 1) {
    count = 1;
  }
  return count;
#endif
}

#if !defined(__APPLE__)  // Linux or Android
std::string get_cpu_name() {
  FILE *fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return "";
  }
  char line[1024];
  while (!feof(fp)) {
    char *s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "Hardware") != NULL) {
      fclose(fp);
      return std::string(line);
    }
  }
  fclose(fp);
  return "";
}

int get_cpu_max_freq_khz(int cpu_id) {
  // first try, for all possible cpu
  char path[256];
#ifdef __ANDROID__
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpu_id);
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    // second try, for online cpu
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpu_id);
    fp = fopen(path, "rb");
    if (!fp) {
      // third try, for online cpu
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
               cpu_id);
      fp = fopen(path, "rb");
      if (!fp) {
        return 0;
      }
      int max_freq_khz = 0;
      if (fscanf(fp, "%d", &max_freq_khz) <= 0) {
        max_freq_khz = 0;
      }
      fclose(fp);
      return max_freq_khz;
    }
  }
  int max_freq_khz = 0;
  while (!feof(fp)) {
    int freq_khz = 0;
    int nscan = fscanf(fp, "%d %*d", &freq_khz);
    if (nscan != 1) {
      break;
    }
    if (freq_khz > max_freq_khz) {
      max_freq_khz = freq_khz;
    }
  }
  fclose(fp);
  return max_freq_khz;
#else
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu_id);
  FILE *fp = fopen(path, "r");
  if (!fp) {
    return 0;
  }
  int max_freq_khz = 0;
  if (fscanf(fp, "%d", &max_freq_khz) <= 0) {
    max_freq_khz = 0;
  }
  fclose(fp);
  return max_freq_khz;
#endif
}

void get_cpu_cache_size(int cpu_id, int *l1_cache_size, int *l2_cache_size,
                        int *l3_cache_size) {
  int max_cache_idx_num = 10;
  *l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  *l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  *l3_cache_size = DEFAULT_L3_CACHE_SIZE;
  for (int i = 0; i < max_cache_idx_num; i++) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu_id, i);
    FILE *fp = fopen(path, "rb");
    if (fp) {
      int level = -1;
      fscanf(fp, "%d", &level);
      fclose(fp);
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu_id, i);
      fp = fopen(path, "rb");
      if (fp) {
        int size = -1;
        fscanf(fp, "%d", &size);
        fclose(fp);
        if (size >= 0) {
          if (level == 1) {
            *l1_cache_size = size * 1024;
          } else if (level == 2) {
            *l2_cache_size = size * 1024;
          } else if (level == 3) {
            *l3_cache_size = size * 1024;
          }
        }
      }
    }
  }
}

int check_online(std::vector<int> *cpu_ids) {
  if (cpu_ids->size() == 0) {
    return 0;
  }
  std::vector<int> online_cpu_ids;
  char path[256];
  for (int i = 0; i < cpu_ids->size(); i++) {
    int cpu_id = (*cpu_ids)[i];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online",
             cpu_id);
    FILE *fp = fopen(path, "rb");
    if (fp) {
      int is_online = 0;
      fscanf(fp, "%d", &is_online);
      fclose(fp);
      if (is_online != 0) {
        online_cpu_ids.push_back(cpu_id);
      }
    }
    // open failed(Permission denied)
  }
  *cpu_ids = online_cpu_ids;
  return cpu_ids->size();
}

int set_sched_affinity(const std::vector<int> &cpu_ids) {
// cpu_set_t definition
// ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long))  // NOLINT
  typedef struct {
    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];  // NOLINT
  } cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

  // set affinity for thread
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < cpu_ids.size(); i++) {
    CPU_SET(cpu_ids[i], &mask);
  }
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    LOG(kLOG_WARNING) << "invoke syscall(__NR_sched_setaffinity) error(ret="
                      << syscallret << ")";
    return -1;
  }
  return 0;
}

int get_cpu_info_by_name(int *cpu_num, ARMArch *arch,
                         std::vector<int> *big_core_ids,
                         std::vector<int> *little_core_ids,
                         std::vector<int> *l1_cache_sizes,
                         std::vector<int> *l2_cache_sizes,
                         std::vector<int> *l3_cache_sizes,
                         std::string hardware_name) {
  /* Snapdragon */
  if (hardware_name.find("SDM845") != std::string::npos) {  // 845
    *cpu_num = 8;
    *arch = A75;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    l1_cache_sizes->resize(*cpu_num);
    l2_cache_sizes->resize(*cpu_num);
    l3_cache_sizes->resize(*cpu_num);
    fill_cpu_cache_size(l1_cache_sizes, 64 * 1024);
    fill_cpu_cache_size(l2_cache_sizes, 256 * 1024, *big_core_ids);
    fill_cpu_cache_size(l2_cache_sizes, 128 * 1024, *little_core_ids);
    fill_cpu_cache_size(l3_cache_sizes, 2048 * 1024);
    return 0;
  } else if (hardware_name.find("SDM710") != std::string::npos) {  // 710
    *cpu_num = 8;
    *arch = A75;
    *big_core_ids = {6, 7};
    *little_core_ids = {0, 1, 2, 3, 4, 5};
    l1_cache_sizes->resize(*cpu_num);
    l2_cache_sizes->resize(*cpu_num);
    l3_cache_sizes->resize(*cpu_num);
    fill_cpu_cache_size(l1_cache_sizes, 64 * 1024, *big_core_ids);
    fill_cpu_cache_size(l1_cache_sizes, 32 * 1024, *little_core_ids);
    fill_cpu_cache_size(l2_cache_sizes, 256 * 1024, *big_core_ids);
    fill_cpu_cache_size(l2_cache_sizes, 128 * 1024, *little_core_ids);
    fill_cpu_cache_size(l3_cache_sizes, 1024 * 1024);
    return 0;
  } else if (hardware_name.find("MSM8998") != std::string::npos) {  // 835
    *cpu_num = 8;
    *arch = A73;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    l1_cache_sizes->resize(*cpu_num);
    l2_cache_sizes->resize(*cpu_num);
    l3_cache_sizes->resize(*cpu_num);
    fill_cpu_cache_size(l1_cache_sizes, 64 * 1024, *big_core_ids);
    fill_cpu_cache_size(l1_cache_sizes, 32 * 1024, *little_core_ids);
    // real L2 cache size is 2M, while that will get bad performace on conv3x3s1
    // or gemm, set to 1M or 512K
    // fill_cpu_cache_size(l2_cache_sizes, 2048 *1024,
    // *big_core_ids);
    // fill_cpu_cache_size(l2_cache_sizes, 1024 * 1024,
    // *little_core_ids);
    fill_cpu_cache_size(l2_cache_sizes, 1024 * 1024);
    fill_cpu_cache_size(l3_cache_sizes, 0);
    return 0;
  } else if (hardware_name.find("MSM8976") != std::string::npos) {  // 652,653
    *cpu_num = 8;
    *arch = A72;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    l1_cache_sizes->resize(*cpu_num);
    l2_cache_sizes->resize(*cpu_num);
    l3_cache_sizes->resize(*cpu_num);
    fill_cpu_cache_size(l1_cache_sizes, 32 * 1024);
    fill_cpu_cache_size(l2_cache_sizes, 1024 * 1024);
    fill_cpu_cache_size(l3_cache_sizes, 0);
    return 0;
  } else if (hardware_name.find("SDM660") != std::string::npos ||
             hardware_name.find("SDM636") != std::string::npos) {  // 660, 636
    *cpu_num = 8;
    *arch = A73;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    l1_cache_sizes->resize(*cpu_num);
    l2_cache_sizes->resize(*cpu_num);
    l3_cache_sizes->resize(*cpu_num);
    fill_cpu_cache_size(l1_cache_sizes, 64 * 1024);
    fill_cpu_cache_size(l2_cache_sizes, 1024 * 1024);
    fill_cpu_cache_size(l3_cache_sizes, 0);
    return 0;

    /* MediaTek */
  } else if (hardware_name.find("MT6799") != std::string::npos) {  // X30
    *cpu_num = 10;
    *arch = A73;
    *big_core_ids = {8, 9};
    *little_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    return 0;
  } else if (hardware_name.find("MT6771") != std::string::npos) {  // P60
    *cpu_num = 8;
    *arch = A73;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    return 0;

    /* Kirin */
  } else if (hardware_name.find("KIRIN970") !=
             std::string::npos) {  // Kirin 970
    *cpu_num = 8;
    *arch = A73;
    *big_core_ids = {4, 5, 6, 7};
    *little_core_ids = {0, 1, 2, 3};
    return 0;
  }
  return -1;
}

// divide cpu cores into big and little clusters by max frequency
void get_cpu_info_by_probe(int cpu_num, std::vector<int> *big_core_ids,
                           std::vector<int> *little_core_ids,
                           std::vector<int> *l1_cache_sizes,
                           std::vector<int> *l2_cache_sizes,
                           std::vector<int> *l3_cache_sizes) {
  // get maxium & minium of cpu_max_freqs
  std::vector<int> cpu_max_freqs(cpu_num);
  for (int i = 0; i < cpu_num; i++) {
    cpu_max_freqs[i] = get_cpu_max_freq_khz(i) / 1000;
  }
  int max_cpu_max_freq = cpu_max_freqs[0];
  int min_cpu_max_freq = cpu_max_freqs[0];
  for (int i = 1; i < cpu_num; i++) {
    int cur_cpu_max_freq = cpu_max_freqs[i];
    if (cur_cpu_max_freq < min_cpu_max_freq) {
      min_cpu_max_freq = cur_cpu_max_freq;
    } else if (cur_cpu_max_freq > max_cpu_max_freq) {
      max_cpu_max_freq = cur_cpu_max_freq;
    }
  }
  int mid_max_freq_khz = (max_cpu_max_freq + min_cpu_max_freq) / 2;
  big_core_ids->clear();
  little_core_ids->clear();
  for (int i = 0; i < cpu_num; i++) {
    if (cpu_max_freqs[i] >= mid_max_freq_khz) {
      big_core_ids->push_back(i);
    } else {
      little_core_ids->push_back(i);
    }
  }
  /* get l1, l2, l3 cache size for each core */
  l1_cache_sizes->resize(cpu_num);
  l2_cache_sizes->resize(cpu_num);
  l3_cache_sizes->resize(cpu_num);
  for (int i = 0; i < cpu_num; i++) {
    get_cpu_cache_size(i, &((*l1_cache_sizes)[i]), &((*l2_cache_sizes)[i]),
                       &((*l3_cache_sizes)[i]));
  }
}

void bind_threads(const std::vector<int> &cpu_ids) {
#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
  std::vector<int> ssarets;
  for (int i = 0; i < num_threads; i++) {
    ssarets.push_back(0);
  }
#pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    ssarets[i] = set_sched_affinity(cpu_ids);
  }
  for (int i = 0; i < num_threads; i++) {
    if (ssarets[i] != 0) {
      LOG(kLOG_WARNING) << "set cpu affinity failed, thread idx: " << i;
      return;
    }
  }
#else
  int ssaret = set_sched_affinity(cpu_ids);
  if (ssaret != 0) {
    LOG(kLOG_WARNING) << "set cpu affinity failed, thread idx: 0 ";
    return;
  }
#endif
}
#endif

CPUContext::CPUContext() {
  _cpu_num = get_cpu_num();
  _big_core_ids.clear();
  _little_core_ids.clear();
#ifdef __APPLE__
  // set default L1, L2 and L3 cache sizes
  _l1_cache_sizes.resize(_cpu_num);
  _l2_cache_sizes.resize(_cpu_num);
  _l3_cache_sizes.resize(_cpu_num);
  fill_cpu_cache_size(&_l1_cache_sizes, DEFAULT_L1_CACHE_SIZE);
  fill_cpu_cache_size(&_l2_cache_sizes, DEFAULT_L2_CACHE_SIZE);
  fill_cpu_cache_size(&_l3_cache_sizes, DEFAULT_L3_CACHE_SIZE);
#else  // Linux or Android
  // probe cpu info, and set big&litte clusters, L1, L2 and L3 cache sizes
  std::string cpu_name = get_cpu_name();
  bool failed =
      get_cpu_info_by_name(&_cpu_num, &_arch, &_big_core_ids, &_little_core_ids,
                           &_l1_cache_sizes, &_l2_cache_sizes, &_l3_cache_sizes,
                           cpu_name) != 0;
  if (failed) {
    get_cpu_info_by_probe(_cpu_num, &_big_core_ids, &_little_core_ids,
                          &_l1_cache_sizes, &_l2_cache_sizes, &_l3_cache_sizes);
  }
  LOG(kLOG_INFO) << "CPU num: " << _cpu_num;
  for (int i = 0; i < _cpu_num; i++) {
    if (!(_l1_cache_sizes.size() > i && _l2_cache_sizes.size() > i &&
          _l3_cache_sizes.size() > i)) {
      break;
    }
    LOG(kLOG_INFO) << i << " L1 Cache: " << _l1_cache_sizes[i] << "KB"
                   << " L2 Cache: " << _l2_cache_sizes[i] << "KB"
                   << " L3 Cache: " << _l3_cache_sizes[i] << "KB";
  }
  LOG(kLOG_INFO) << "Big cores: ";
  for (int i = 0; i < _big_core_ids.size(); i++) {
    LOG(kLOG_INFO) << _big_core_ids[i];
  }
  LOG(kLOG_INFO) << "Little cores: ";
  for (int i = 0; i < _little_core_ids.size(); i++) {
    LOG(kLOG_INFO) << _little_core_ids[i];
  }
#endif
  // use single thread by default
  set_thread_num(1, PERFORMANCE_PRIORITY);
}

void CPUContext::set_thread_num(int thread_num, PowerMode power_mode) {
  int big_core_num = _big_core_ids.size();
  int little_core_num = _little_core_ids.size();
#ifdef _OPENMP
  if (thread_num > _cpu_num) {
    thread_num = _cpu_num;
  }
#else
  thread_num = 1;
#endif
  std::vector<int> bind_core_ids;
  if (power_mode == PERFORMANCE_PRIORITY || power_mode == PERFORMANCE_ONLY) {
    if (big_core_num > 0) {
      bind_core_ids = _big_core_ids;
      if (power_mode == PERFORMANCE_ONLY && thread_num > big_core_num) {
        LOG(kLOG_ERROR) << "thread_num(" << thread_num
                        << ") exceed the big cores num (" << big_core_num << ")"
                        << ", force to set thread_num = " << big_core_num;
        thread_num = big_core_num;
      }
    }
  } else if (power_mode == EFFICIENCY_PRIORITY ||
             power_mode == EFFICIENCY_ONLY) {
    if (little_core_num > 0) {
      bind_core_ids = _little_core_ids;
      if (power_mode == EFFICIENCY_ONLY && thread_num > little_core_num) {
        LOG(kLOG_ERROR) << "thread_num(" << thread_num
                        << ") exceed the little cores num (" << little_core_num
                        << ")"
                        << ", force to set thread_num = " << little_core_num;
        thread_num = little_core_num;
      }
    }
  }
  _power_mode = AUTO;
#ifdef _OPENMP
  omp_set_num_threads(thread_num);
  thread_num = omp_get_max_threads();
#endif
#if !defined(__APPLE__)  // Linux or Android
  if (bind_core_ids.size() > 0 && check_online(&bind_core_ids) >= thread_num) {
    bind_threads(bind_core_ids);
    _power_mode = power_mode;
  }
#endif
  LOG(kLOG_INFO) << "thread num: " << thread_num
                 << " power mode: " << _power_mode;
}

int CPUContext::get_thread_num() {
  int thread_num = 1;
#ifdef _OPENMP
  thread_num = omp_get_max_threads();
#endif
  return thread_num;
}

int CPUContext::get_cache_size(int level) {
  std::vector<int> *ptr = nullptr;
  if (level == 1) {
    ptr = &_l1_cache_sizes;
  } else if (level == 2) {
    ptr = &_l2_cache_sizes;
  } else if (level == 3) {
    ptr = &_l3_cache_sizes;
  } else {
    return 0;
  }
  if (_power_mode == PERFORMANCE_PRIORITY || _power_mode == PERFORMANCE_ONLY) {
    if (_big_core_ids.size() > 0) {
      int idx = _big_core_ids[0];
      if (ptr->size() > idx) {
        return (*ptr)[idx];
      }
    }
  } else if (_power_mode == EFFICIENCY_PRIORITY ||
             _power_mode == EFFICIENCY_ONLY) {
    if (_little_core_ids.size() > 0) {
      int idx = _little_core_ids[0];
      if (ptr->size() > idx) {
        return (*ptr)[idx];
      }
    }
  } else {  // AUTO
    int idx = 0;
    if (ptr->size() > idx) {
      return (*ptr)[idx];
    }
  }
}

void *CPUContext::get_work_space(int size_in_byte) {
  return reinterpret_cast<void *>(
      _workspace.mutable_data<int8_t>(make_ddim({size_in_byte})));
}

}  // namespace framework
}  // namespace paddle_mobile
