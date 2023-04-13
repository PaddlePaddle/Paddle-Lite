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

// Parts of the following code in this file refs to
// https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
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

// Parts of the following code in this file refs to
// https://github.com/mjp9527/MegEngine/blob/master/dnn/src/x86/utils.cpp
/**
 * \file dnn/src/x86/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "utilities/cpu_info.h"
#include <stdarg.h>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "utilities/logging.h"
#ifdef ARM_DNN_LIBRARY_WITH_OMP
#include <omp.h>
#endif

#if ARM_DNN_LIBRARY_OS_LINUX
#include <sys/syscall.h>
#include <unistd.h>
// http://elixir.free-electrons.com/linux/latest/source/arch/arm64/include/uapi/asm/hwcap.h
#include <asm/hwcap.h> /* Get HWCAP bits from asm/hwcap.h */
#include <sys/auxv.h>
#if ARM_DNN_LIBRARY_OS_ANDROID
#include <sys/system_properties.h>
#endif  // ARM_DNN_LIBRARY_OS_ANDROID
#elif ARM_DNN_LIBRARY_OS_IOS || ARM_DNN_LIBRARY_OS_MAC
#include "TargetConditionals.h"
#if ARM_DNN_LIBRARY_OS_IOS
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // ARM_DNN_LIBRARY_OS_IOS
#elif ARM_DNN_LIBRARY_OS_QNX
#include <sys/neutrino.h>
#endif  // ARM_DNN_LIBRARY_OS_LINUX

namespace armdnnlibrary {

ARM_DNN_LIBRARY_THREAD_LOCAL PowerMode CPUInfo::power_mode_ =
    PowerMode::NO_BIND;
ARM_DNN_LIBRARY_THREAD_LOCAL std::vector<int> CPUInfo::active_idxs_ = {0};

void set_cpu_attrs(std::vector<CPUAttr>* cpu_attrs,
                   int from_cpu_id,
                   int to_cpu_id,
                   int argc,
                   ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  ARM_DNN_LIBRARY_CHECK(cpu_attrs);
  auto cpu_num = cpu_attrs->size();
  ARM_DNN_LIBRARY_CHECK_GT(cpu_num, 0);
  ARM_DNN_LIBRARY_CHECK_GE(from_cpu_id, 0);
  ARM_DNN_LIBRARY_CHECK_GT(to_cpu_id, 0);
  ARM_DNN_LIBRARY_CHECK_LT(from_cpu_id, cpu_num);
  ARM_DNN_LIBRARY_CHECK_LE(to_cpu_id, cpu_num);
  for (size_t i = 0; i < argc; i++) {
    int attr_index = va_arg(arg_ptr, int32_t);
    switch (attr_index) {
      case CPU_ATTR_ARCH: {
        CPUArch attr_value = static_cast<CPUArch>(va_arg(arg_ptr, int32_t));
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).arch = attr_value;
        }
      } break;
      case CPU_ATTR_CLUSTER_ID: {
        int32_t attr_value = va_arg(arg_ptr, int32_t);
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).cluster_id = attr_value;
        }
      } break;
      case CPU_ATTR_L1_CACHE_SIZE: {
        int32_t attr_value = va_arg(arg_ptr, int32_t);
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).l1_cache_size = attr_value;
        }
      } break;
      case CPU_ATTR_L2_CACHE_SIZE: {
        int32_t attr_value = va_arg(arg_ptr, int32_t);
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).l2_cache_size = attr_value;
        }
      } break;
      case CPU_ATTR_L3_CACHE_SIZE: {
        int32_t attr_value = va_arg(arg_ptr, int32_t);
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).l3_cache_size = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_FP16: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_fp16 = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_BF16: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_bf16 = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_DOTPROD: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_dotprod = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_SVE2: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_sve2 = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_SVE2_I8MM: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_sve2_i8mm = attr_value;
        }
      } break;
      case CPU_ATTR_SUPPORT_ARM_SVE2_F32MM: {
        bool attr_value = va_arg(arg_ptr, int32_t) != 0;
        for (size_t j = from_cpu_id; j < to_cpu_id; j++) {
          cpu_attrs->at(j).support_arm_sve2_f32mm = attr_value;
        }
      } break;
      default:
        break;
    }
  }
  va_end(arg_ptr);
}

bool get_cluster_ids_by_cpu_max_freqs(const std::vector<int>& cpu_max_freqs,
                                      std::vector<int>* cluster_ids) {
  size_t cpu_num = cpu_max_freqs.size();
  if (cpu_num == 0) return false;
  std::map<int, std::vector<int>> cluster_freqs;
  for (size_t i = 0; i < cpu_num; i++) {
    auto key = cpu_max_freqs[i];
    auto it = cluster_freqs.find(key);
    if (it == cluster_freqs.end()) {
      auto result =
          cluster_freqs.insert(std::make_pair(key, std::vector<int>()));
      it = result.first;
    }
    it->second.push_back(i);
  }
  // LITTLE
  auto cluster_count = cluster_freqs.size();
  ARM_DNN_LIBRARY_CHECK_GT(cluster_count, 0);
  cluster_ids->resize(cpu_num);
  for (size_t i = 0; i < cpu_num; i++) {
    cluster_ids->at(i) = 0;
  }
  if (cluster_count > 1) {
    // Middle
    for (size_t i = 0; i < cpu_num; i++) {
      cluster_ids->at(i) = 1;
    }
    // LITTLE
    auto cpu_ids = cluster_freqs.begin()->second;
    for (size_t i = 0; i < cpu_ids.size(); i++) {
      cluster_ids->at(cpu_ids[i]) = 0;
    }
    // big
    cpu_ids = cluster_freqs.rbegin()->second;
    for (size_t i = 0; i < cpu_ids.size(); i++) {
      cluster_ids->at(cpu_ids[i]) = 2;
    }
  }
  return true;
}

// Initialize CPUInfo on Linux or Android
#if ARM_DNN_LIBRARY_OS_LINUX
std::string get_cpu_name() {
  std::string cpu_name = "";
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return "";
  }
  char line[1024];
  bool first_model_name = true;
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "Hardware") != NULL) {
      cpu_name += std::string(line);
    }
    if (strstr(line, "model name") != NULL && first_model_name) {
      cpu_name += std::string(line);
      first_model_name = false;
    }
  }
#if ARM_DNN_LIBRARY_OS_ANDROID
  // cpu name concat board name, platform name and chip name
  char board_name[128];
  char platform_name[128];
  char chip_name[128];
  __system_property_get("ro.product.board", board_name);
  __system_property_get("ro.board.platform", platform_name);
  __system_property_get("ro.chipname", chip_name);
  cpu_name =
      cpu_name + "_" + board_name + "_" + platform_name + "_" + chip_name;
#endif  // ARM_DNN_LIBRARY_OS_ANDROID
  std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
  fclose(fp);
  return cpu_name;
}

int get_cpu_num() {
  int cpu_num = 0;
  while (true) {
    char path[256];
    snprintf(
        path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", cpu_num);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
      break;
    }
    cpu_num++;
    fclose(fp);
  }
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
}

void get_cpu_cache_size(int cpu_idx,
                        size_t* l1_cache_size,
                        size_t* l2_cache_size,
                        size_t* l3_cache_size) {
  *l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  *l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  *l3_cache_size = DEFAULT_L3_CACHE_SIZE;
  int max_cache_idx_num = 10;
  for (int i = 0; i < max_cache_idx_num; i++) {
    char path[256];
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level",
             cpu_idx,
             i);
    FILE* fp = fopen(path, "rb");
    if (fp) {
      int level = -1;
      fscanf(fp, "%d", &level);
      fclose(fp);
      snprintf(path,
               sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cache/index%d/size",
               cpu_idx,
               i);
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

int get_cpu_min_freq_khz(int cpu_idx) {
  int min_freq_khz = 0;
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_min_freq",
           cpu_idx);
  FILE* fp = fopen(path, "rb");
  if (fp) {
    fscanf(fp, "%d", &min_freq_khz);
    fclose(fp);
  }
  if (min_freq_khz <= 0) {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "cpuinfo_min_freq should be greater than 0, but receive "
        << min_freq_khz;
  }
  return min_freq_khz;
}

int get_cpu_max_freq_khz(int cpu_idx) {
  int max_freq_khz = 0;
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
           cpu_idx);
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpu_idx);
    fp = fopen(path, "rb");
  }
  if (fp) {
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
  }
  if (max_freq_khz <= 0) {
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
             cpu_idx);
    fp = fopen(path, "rb");
    if (fp) {
      fscanf(fp, "%d", &max_freq_khz);
      fclose(fp);
    }
  }
  if (max_freq_khz <= 0) {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "cpuinfo_max_freq should be greater than 0, but receive "
        << max_freq_khz;
  }
  return max_freq_khz;
}

bool get_cpu_online(const std::vector<int>& cpu_idxs) {
  auto cpu_num = cpu_idxs.size();
  if (cpu_num == 0) {
    return false;
  }
  bool all_online = true;
  FILE* fp = fopen("/sys/devices/system/cpu/online", "r");
  if (fp) {
    std::vector<int> online_idxs;
    int cnt, idx;
    char sep;
    int prev = -1;
    for (;;) {
      cnt = fscanf(fp, "%u%c", &idx, &sep);
      if (cnt <= 0) break;
      if (prev >= 0) {
        while (++prev < idx) online_idxs.push_back(prev);
      }
      online_idxs.push_back(idx);
      if (cnt == 2 && sep == '-')
        prev = idx;
      else
        prev = -1;
      if (cnt == 1 || sep == '\n') break;
    }
    for (size_t i = 0; i < cpu_num; i++) {
      if (std::find(online_idxs.begin(), online_idxs.end(), cpu_idxs[i]) ==
          online_idxs.end()) {
        all_online = false;
        ARM_DNN_LIBRARY_LOG(WARNING) << "CPU idx: " << cpu_idxs[i]
                                     << " is offine";
      }
    }
    fclose(fp);
  } else {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "Failed to query all cpus online status list. Try to "
           "query single cpu online status.";
    char path[256];
    for (int i = 0; i < cpu_idxs.size(); i++) {
      snprintf(path,
               sizeof(path),
               "/sys/devices/system/cpu/cpu%d/online",
               cpu_idxs[i]);
      FILE* fp = fopen(path, "rb");
      int is_online = 0;
      if (fp) {
        fscanf(fp, "%d", &is_online);
        fclose(fp);
      } else {
        ARM_DNN_LIBRARY_LOG(ERROR)
            << "Failed to query the online status of CPU idx: " << cpu_idxs[i];
      }
      if (is_online == 0) {
        all_online = false;
        ARM_DNN_LIBRARY_LOG(WARNING) << "CPU idx: " << cpu_idxs[i]
                                     << " is offine.";
      }
    }
  }
  return all_online;
}

bool set_cpu_sched_affinity(const std::vector<int>& cpu_idxs) {
#define PD_CPU_SETSIZE 1024
#define PD__NCPUBITS (8 * sizeof(unsigned long))  // NOLINT
  typedef struct {
    unsigned long __bits[PD_CPU_SETSIZE / PD__NCPUBITS];  // NOLINT
  } cpu_set_t;
#define PD_CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu) / PD__NCPUBITS] |= (1UL << ((cpu) % PD__NCPUBITS)))
#define PD_CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif  // __GLIBC__
  cpu_set_t mask;
  PD_CPU_ZERO(&mask);
  unsigned int RUNMASK = 0;
  for (int i = 0; i < cpu_idxs.size(); ++i) {
    PD_CPU_SET(cpu_idxs[i], &mask);
  }
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  return syscallret != 0;
}

bool bind_threads_to_cpus(const std::vector<int> cpu_idxs) {
  if (!get_cpu_online(cpu_idxs)) {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "Failed to bind threads to CPUs, because some cores are offline.";
    return false;
  }
#ifdef ARM_DNN_LIBRARY_WITH_OMP
  int thread_num = cpu_idxs.size();
  omp_set_num_threads(thread_num);
  std::vector<int> ssarets(thread_num, 0);
#pragma omp parallel for
  for (int i = 0; i < thread_num; i++) {
    ssarets[i] = set_cpu_sched_affinity(cpu_idxs);
  }
  for (int i = 0; i < thread_num; i++) {
    if (ssarets[i] != 0) {
      ARM_DNN_LIBRARY_LOG(WARNING) << "Set cpu affinity failed, core idx: "
                                   << cpu_idxs[i];
      return false;
    }
  }
#else   // ARM_DNN_LIBRARY_WITH_OMP
  std::vector<int> first_cpu_idx;
  first_cpu_idx.push_back(cpu_idxs[0]);
  int ssaret = set_cpu_sched_affinity(first_cpu_idx);
  if (ssaret != 0) {
    ARM_DNN_LIBRARY_LOG(WARNING) << "Set cpu affinity failed, core idx: "
                                 << cpu_idxs[0];
    return false;
  }
#endif  // ARM_DNN_LIBRARY_WITH_OMP
  return true;
}

bool detect_from_cpu_info(std::vector<CPUAttr>* cpu_attrs) {
  cpu_attrs->clear();
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) return false;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) break;
    if (strstr(line, "part") != NULL) {
      CPUAttr cpu_attr;
      int cpu_part = 0;
      sscanf(s, "CPU part\t: %x", &cpu_part);
      switch (cpu_part) {
        case 0xd03:
          cpu_attr.arch = CPUArch::CORTEX_A53;
          break;
        case 0xd04:
          cpu_attr.arch = CPUArch::CORTEX_A35;
          break;
        case 0x803:
        case 0x805:
        case 0xd05:
          cpu_attr.arch = CPUArch::CORTEX_A55;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd07:
          cpu_attr.arch = CPUArch::CORTEX_A57;
          break;
        case 0xd08:
        case 0x205:
          cpu_attr.arch = CPUArch::CORTEX_A72;
          break;
        case 0x800:
        case 0x801:
        case 0xd09:
          cpu_attr.arch = CPUArch::CORTEX_A73;
          break;
        case 0x802:
        case 0xd0a:
          cpu_attr.arch = CPUArch::CORTEX_A75;
          cpu_attr.support_arm_fp16 = true;
          break;
        case 0x804:
        case 0xd40:
          cpu_attr.arch = CPUArch::CORTEX_A76;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd0d:
          cpu_attr.arch = CPUArch::CORTEX_A77;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd41:
          // 888
          cpu_attr.arch = CPUArch::CORTEX_A78;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd44:
          // 888
          cpu_attr.arch = CPUArch::CORTEX_X1;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd46:
          cpu_attr.arch = CPUArch::CORTEX_A510;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd47:
          cpu_attr.arch = CPUArch::CORTEX_A710;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        case 0xd48:
          cpu_attr.arch = CPUArch::CORTEX_X2;
          cpu_attr.support_arm_fp16 = true;
          cpu_attr.support_arm_dotprod = true;
          break;
        default:
          ARM_DNN_LIBRARY_LOG(WARNING) << "Unknow cpu part: 0x" << std::hex
                                       << cpu_part;
          break;
      }
      cpu_attrs->push_back(cpu_attr);
    }
  }
  fclose(fp);
  auto cpu_num = cpu_attrs->size();
// Get sve2, sve2+i8mm, sve2+f32mm support from the AT HWCAP2 auxillary vector
// entry.
#if ARM_DNN_LIBRARY_OS_ANDROID && ARM_DNN_LIBRARY_ARCH_ARM64
#define AT_HWCAP2 26
#define AARCH64_HWCAP2_SVE2 (1UL << 1)
#define AARCH64_HWCAP2_SVEI8MM (1UL << 9)
#define AARCH64_HWCAP2_SVEF32MM (1UL << 10)
  {
    auto hwcap2_mask =
        static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
    bool support_arm_sve2 = hwcap2_mask & AARCH64_HWCAP2_SVE2;
    bool support_arm_sve2_i8mm = hwcap2_mask & AARCH64_HWCAP2_SVEI8MM;
    bool support_arm_sve2_f32mm = hwcap2_mask & AARCH64_HWCAP2_SVEF32MM;
    for (size_t i = 0; i < cpu_num; i++) {
      cpu_attrs->at(i).support_arm_sve2 = support_arm_sve2;
      cpu_attrs->at(i).support_arm_sve2_i8mm = support_arm_sve2_i8mm;
      cpu_attrs->at(i).support_arm_sve2_f32mm = support_arm_sve2_f32mm;
    }
  }
#undef AARCH64_HWCAP2_SVEF32MM
#undef AARCH64_HWCAP2_SVEI8MM
#undef AARCH64_HWCAP2_SVE2
#undef AT_HWCAP2
#endif
  // Get cluster id by max frequency
  std::vector<int> cpu_max_freqs(cpu_num, 0);
  for (size_t i = 0; i < cpu_num; i++) {
    cpu_max_freqs[i] = get_cpu_max_freq_khz(i);
  }
  std::vector<int> cluster_ids(cpu_num, 0);
  get_cluster_ids_by_cpu_max_freqs(cpu_max_freqs, &cluster_ids);
  for (size_t i = 0; i < cpu_num; i++) {
    cpu_attrs->at(i).cluster_id = cluster_ids[i];
  }
  // Get L1, L2, L3 cache size
  for (size_t i = 0; i < cpu_num; i++) {
    get_cpu_cache_size(i,
                       &cpu_attrs->at(i).l1_cache_size,
                       &cpu_attrs->at(i).l2_cache_size,
                       &cpu_attrs->at(i).l3_cache_size);
  }
  return true;
}

bool detect_from_cpu_name(std::vector<CPUAttr>* cpu_attrs) {
  const std::string& cpu_name = get_cpu_name();
  ARM_DNN_LIBRARY_LOG(INFO) << cpu_name;
  if (cpu_name.length() == 0) return false;
  if (cpu_name.find("SM8350") != std::string::npos) {  // 888
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0 /* from_cpu_id */,
                  4 /* to_cpu_id */,
                  4 /* attrs count */,
                  CPU_ATTR_ARCH /* arch */,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID /* cluster_id */,
                  0 /* LITTLE */,
                  CPU_ATTR_L1_CACHE_SIZE /* l1_cache_size */,
                  256 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE /* l2_cache_size */,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  7,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A78,
                  CPU_ATTR_CLUSTER_ID,
                  1 /* Middle */,
                  CPU_ATTR_L1_CACHE_SIZE,
                  192 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  7,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_X1,
                  CPU_ATTR_CLUSTER_ID,
                  2 /* big */,
                  CPU_ATTR_L1_CACHE_SIZE,
                  512 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE /* l3_cache_size */,
                  4 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16 /* support_arm_fp16 */,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD /* support_arm_dotprod */,
                  1);
  } else if (cpu_name.find("SA8155") != std::string::npos) {  // SA8155
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::KRYO_485_SILVER,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  128 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  7,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::KRYO_485_GOLD,
                  CPU_ATTR_CLUSTER_ID,
                  1,
                  CPU_ATTR_L1_CACHE_SIZE,
                  256 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  256 * 1024);
    set_cpu_attrs(cpu_attrs,
                  7,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::KRYO_485_GOLD_PRIME,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  512 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  2 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("SA8195") != std::string::npos) {  // SA8195
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::KRYO_485_SILVER,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  128 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::KRYO_485_GOLD_PRIME,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  512 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  4 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("KONA") != std::string::npos) {  // 865
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  256 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A77,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  192 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  768 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  4 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("SM8150") != std::string::npos) {  // 855
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A76,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  256 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  2 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("SDM845") != std::string::npos) {  // 845
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A75,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  256 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_L3_CACHE_SIZE,
                  2 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1);
  } else if (cpu_name.find("SDM710") != std::string::npos) {  // 710
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  6,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  6,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A75,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  256 * 1024);
    set_cpu_attrs(cpu_attrs, 0, 8, 1, CPU_ATTR_L3_CACHE_SIZE, 1024 * 1024);
  } else if (cpu_name.find("MSM8998") != std::string::npos) {  // 835
    cpu_attrs->resize(8);
    set_cpu_attrs(
        cpu_attrs,
        0,
        4,
        4,
        CPU_ATTR_ARCH,
        CPUArch::CORTEX_A53,
        CPU_ATTR_CLUSTER_ID,
        0,
        CPU_ATTR_L1_CACHE_SIZE,
        32 * 1024,
        CPU_ATTR_L2_CACHE_SIZE,
        /* Real cache size is 2MB, while that will get bad performace on
           conv3x3s1 or gemm, set to 1MB or 512KB */
        1024 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A73,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
  } else if (cpu_name.find("MSM8996") != std::string::npos) {  // 820
    cpu_attrs->resize(4);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A72,
                  CPU_ATTR_L1_CACHE_SIZE,
                  24 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  2,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  2,
                  4,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
  } else if (cpu_name.find("SDM660") != std::string::npos ||
             cpu_name.find("SDM636") != std::string::npos) {  // 660, 636
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A73,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024);
  } else if (cpu_name.find("MSM8976") != std::string::npos) {  // 652,653
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  3,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  3,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A73,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
    set_cpu_attrs(cpu_attrs, 0, 8, 1, CPU_ATTR_L1_CACHE_SIZE, 32 * 1024);
  } else if (cpu_name.find("MSM8953") != std::string::npos) {  // 625
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
  } else if (cpu_name.find("MSM8939") != std::string::npos) {  // 615
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L2_CACHE_SIZE,
                  256 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  2,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
  } else if (cpu_name.find("MT6891") != std::string::npos) {  // Dimensity 1100
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A78,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  4 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("MT6797") != std::string::npos) {  // X20/X23/X25/X27
    cpu_attrs->resize(10);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  8,
                  10,
                  3,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A72,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L2_CACHE_SIZE,
                  1024 * 1024);
    set_cpu_attrs(cpu_attrs, 0, 8, 1, CPU_ATTR_L1_CACHE_SIZE, 32 * 1024);
  } else if (cpu_name.find("MT6799") != std::string::npos) {  // X30
    cpu_attrs->resize(10);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0);
    set_cpu_attrs(cpu_attrs,
                  8,
                  10,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A73,
                  CPU_ATTR_CLUSTER_ID,
                  2);
  } else if (cpu_name.find("MT6795") != std::string::npos ||
             cpu_name.find("MT6762") != std::string::npos ||
             cpu_name.find("MT6755T") != std::string::npos ||
             cpu_name.find("MT6755S") != std::string::npos ||
             cpu_name.find("MT6753") != std::string::npos ||
             cpu_name.find("MT6752") != std::string::npos ||
             cpu_name.find("MT6750") !=
                 std::string::npos) {  // X10, P22, P15/P18, MT6753,
                                       // MT6752/MT6752M, MT6750
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0);
  } else if (cpu_name.find("MT6758") != std::string::npos ||
             cpu_name.find("MT6757") != std::string::npos ||
             cpu_name.find("MT6763") != std::string::npos ||
             cpu_name.find("MT6755M") != std::string::npos ||
             cpu_name.find("MT6755") !=
                 std::string::npos) {  // P30, P20/P25, P23, P10
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs, 0, 8, 1, CPU_ATTR_ARCH, CPUArch::CORTEX_A53);
    set_cpu_attrs(cpu_attrs, 0, 4, 1, CPU_ATTR_CLUSTER_ID, 0);
    set_cpu_attrs(cpu_attrs, 4, 8, 1, CPU_ATTR_CLUSTER_ID, 2);
  } else if (cpu_name.find("MT6771") != std::string::npos) {  // P60
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A73,
                  CPU_ATTR_CLUSTER_ID,
                  2);
  } else if (cpu_name.find("MT6765") != std::string::npos ||
             cpu_name.find("MT6739") != std::string::npos ||
             cpu_name.find("MT6738") != std::string::npos ||
             cpu_name.find("MT6737") !=
                 std::string::npos) {  // A22, MT6739, MT6738, MT6767
    cpu_attrs->resize(4);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  2,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A53,
                  CPU_ATTR_CLUSTER_ID,
                  0);
  } else if (cpu_name.find("KIRIN980") != std::string::npos ||
             cpu_name.find("KIRIN990") !=
                 std::string::npos) {  // Kirin 980, Kirin 990
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A76,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  3,
                  CPU_ATTR_L3_CACHE_SIZE,
                  4 * 1024 * 1024,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("KIRIN810") != std::string::npos) {
    cpu_attrs->resize(8);
    set_cpu_attrs(cpu_attrs,
                  0,
                  4,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A55,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  32 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  128 * 1024);
    set_cpu_attrs(cpu_attrs,
                  4,
                  8,
                  4,
                  CPU_ATTR_ARCH,
                  CPUArch::CORTEX_A76,
                  CPU_ATTR_CLUSTER_ID,
                  2,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  512 * 1024);
    set_cpu_attrs(cpu_attrs,
                  0,
                  8,
                  2,
                  CPU_ATTR_SUPPORT_ARM_FP16,
                  1,
                  CPU_ATTR_SUPPORT_ARM_DOTPROD,
                  1);
  } else if (cpu_name.find("FT2000PLUS") != std::string::npos) {
    cpu_attrs->resize(64);
    set_cpu_attrs(cpu_attrs,
                  0,
                  64,
                  4,
                  CPU_ATTR_CLUSTER_ID,
                  0,
                  CPU_ATTR_L1_CACHE_SIZE,
                  64 * 1024,
                  CPU_ATTR_L2_CACHE_SIZE,
                  32 * 1024 * 1024,
                  CPU_ATTR_L3_CACHE_SIZE,
                  128 * 1024 * 1024);
  } else {
    return false;
  }
  return true;
}

bool CPUInfo::Initialize() {
  bool status = detect_from_cpu_name(&cpu_attrs_);
  if (!status) {
    status = detect_from_cpu_info(&cpu_attrs_);
    if (!status) {
      cpu_attrs_.resize(1);
      set_cpu_attrs(&cpu_attrs_,
                    0,
                    1,
                    11,
                    CPU_ATTR_ARCH,
                    CPUArch::UNKOWN,
                    CPU_ATTR_CLUSTER_ID,
                    0,
                    CPU_ATTR_L1_CACHE_SIZE,
                    DEFAULT_L1_CACHE_SIZE,
                    CPU_ATTR_L2_CACHE_SIZE,
                    DEFAULT_L2_CACHE_SIZE,
                    CPU_ATTR_L3_CACHE_SIZE,
                    DEFAULT_L3_CACHE_SIZE,
                    CPU_ATTR_SUPPORT_ARM_FP16,
                    0,
                    CPU_ATTR_SUPPORT_ARM_BF16,
                    0,
                    CPU_ATTR_SUPPORT_ARM_DOTPROD,
                    0,
                    CPU_ATTR_SUPPORT_ARM_SVE2,
                    0,
                    CPU_ATTR_SUPPORT_ARM_SVE2_I8MM,
                    0,
                    CPU_ATTR_SUPPORT_ARM_SVE2_F32MM,
                    0);
      return false;
    }
  }
  return true;
}
// Initialize CPUInfo on iOS or macOS
#elif ARM_DNN_LIBRARY_OS_IOS || ARM_DNN_LIBRARY_OS_MAC
int get_cpu_num() {
  int cpu_num = 0;
  size_t size = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &size, NULL, 0);
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
}

bool CPUInfo::Initialize() {
  auto cpu_num = get_cpu_num();
  cpu_attrs_.resize(cpu_num);
  set_cpu_attrs(&cpu_attrs_,
                0,
                cpu_num,
                11,
                CPU_ATTR_ARCH,
                CPUArch::APPLE,
                CPU_ATTR_CLUSTER_ID,
                2,
                CPU_ATTR_L1_CACHE_SIZE,
                DEFAULT_L1_CACHE_SIZE,
                CPU_ATTR_L2_CACHE_SIZE,
                DEFAULT_L2_CACHE_SIZE,
                CPU_ATTR_L3_CACHE_SIZE,
                DEFAULT_L3_CACHE_SIZE,
                CPU_ATTR_SUPPORT_ARM_FP16,
                0,
                CPU_ATTR_SUPPORT_ARM_BF16,
                0,
                CPU_ATTR_SUPPORT_ARM_DOTPROD,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2_I8MM,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2_F32MM,
                0);
  return false;
}
// Initialize CPUInfo on QNX
// #elif ARM_DNN_LIBRARY_OS_QNX
#else
bool CPUInfo::Initialize() {
  size_t cpu_num = 1;  // Only support 1 cpu by default.
  ARM_DNN_LIBRARY_LOG(WARNING)
      << "The number of CPUs is forced to 1 because we cannot "
         "detect the number of CPUs on the current platform.";
  cpu_attrs_.resize(cpu_num);
  set_cpu_attrs(&cpu_attrs_,
                0,
                cpu_num,
                11,
                CPU_ATTR_ARCH,
                CPUArch::UNKOWN,
                CPU_ATTR_CLUSTER_ID,
                2,
                CPU_ATTR_L1_CACHE_SIZE,
                DEFAULT_L1_CACHE_SIZE,
                CPU_ATTR_L2_CACHE_SIZE,
                DEFAULT_L2_CACHE_SIZE,
                CPU_ATTR_L3_CACHE_SIZE,
                DEFAULT_L3_CACHE_SIZE,
                CPU_ATTR_SUPPORT_ARM_FP16,
                0,
                CPU_ATTR_SUPPORT_ARM_BF16,
                0,
                CPU_ATTR_SUPPORT_ARM_DOTPROD,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2_I8MM,
                0,
                CPU_ATTR_SUPPORT_ARM_SVE2_F32MM,
                0);
  return false;
}
#endif  // ARM_DNN_LIBRARY_OS_LINUX

CPUInfo::CPUInfo() {
  Initialize();
  // Only use single thread with NO_BIND mode.
  SetRunMode(PowerMode::NO_BIND, 1);
}

CPUInfo::~CPUInfo() {}

CPUInfo& CPUInfo::Singleton() {
  static auto* cpu_info = new CPUInfo;
  return *cpu_info;
}

const CPUAttr& CPUInfo::at(int index) {
  ARM_DNN_LIBRARY_CHECK_GE(index, 0);
  ARM_DNN_LIBRARY_CHECK_LT(index, cpu_attrs_.size());
  return cpu_attrs_[index];
}

bool CPUInfo::SetRunMode(PowerMode power_mode, size_t thread_num) {
  auto cpu_num = cpu_attrs_.size();
  ARM_DNN_LIBRARY_CHECK_GE(cpu_num, 1);
  ARM_DNN_LIBRARY_CHECK_GE(thread_num, 1);
#if defined(ARM_DNN_LIBRARY_WITH_OMP) || \
    defined(ARM_DNN_LIBRARY_WITH_THREAD_POOL)
  if (thread_num > cpu_num) {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "Apply for " << thread_num << " threads with "
        << power_mode_to_string(power_mode) << " mode, but only " << cpu_num
        << " CPUs found, so shrink to " << cpu_num << " threads.";
    thread_num = std::min(thread_num, cpu_num);
  }
#else
  if (thread_num > 1) {
    ARM_DNN_LIBRARY_LOG(WARNING)
        << "Apply for " << thread_num << " threads with "
        << power_mode_to_string(power_mode)
        << " mode, but OpenMP or ThreadPool is disabled, so "
           "shrink to single thread.";
  }
  thread_num = 1;  // Force thread_num to 1 if OpenMP or ThreadPool is disabled.
#endif
  // Detect Little, big cores from cluster_id.
  std::vector<int> big_idxs, little_idxs;
  for (size_t i = 0; i < cpu_num; i++) {
    if (cpu_attrs_[i].cluster_id == 0) little_idxs.push_back(i);
    if (cpu_attrs_[i].cluster_id == 1) big_idxs.push_back(i);
  }
  for (size_t i = 0; i < cpu_num; i++) {
    if (cpu_attrs_[i].cluster_id == 2) big_idxs.push_back(i);
  }
  auto big_num = big_idxs.size();
  auto little_num = little_idxs.size();
  ARM_DNN_LIBRARY_CHECK_EQ(cpu_num, big_num + little_num);
  active_idxs_.clear();
  bool reset_to_no_bind = true;
#if ARM_DNN_LIBRARY_OS_LINUX
  power_mode_ = power_mode;
  if (power_mode_ == PowerMode::HIGH) {
    if (big_num > 0) {
      if (thread_num > big_num) {
        active_idxs_ = big_idxs;
        ARM_DNN_LIBRARY_LOG(WARNING)
            << "Apply for " << thread_num << " threads with "
            << power_mode_to_string(power_mode)
            << " mode, but exceed the number of the big cores " << big_num
            << ", so shrink to " << big_num << " threads.";
      } else {
        for (size_t i = 0; i < thread_num; i++) {
          active_idxs_.push_back(big_idxs[big_num - 1 - i]);
        }
      }
    } else {
      power_mode_ = PowerMode::LOW;
      ARM_DNN_LIBRARY_LOG(WARNING)
          << "There is no big cores, so switch HIGH to LOW mode.";
      if (thread_num > little_num) {
        active_idxs_ = little_idxs;
        ARM_DNN_LIBRARY_LOG(WARNING)
            << "Apply for " << thread_num
            << " threads with LOW mode, but exceed the number of "
               "the LITTLE cores "
            << little_num << ", so shrink to " << little_num << " threads.";
      } else {
        for (size_t i = 0; i < thread_num; i++) {
          active_idxs_.push_back(little_idxs[i]);
        }
      }
    }
  } else if (power_mode_ == PowerMode::LOW) {
    if (little_num > 0) {
      if (thread_num > little_num) {
        ARM_DNN_LIBRARY_LOG(WARNING)
            << "Apply for " << thread_num << " threads with "
            << power_mode_to_string(power_mode)
            << " mode, but exceed the number of the LITTLE cores " << little_num
            << ", so shrink to " << little_num << " threads.";
        active_idxs_ = little_idxs;
      } else {
        for (size_t i = 0; i < thread_num; i++) {
          active_idxs_.push_back(little_idxs[i]);
        }
      }
    } else {
      power_mode_ = PowerMode::HIGH;
      ARM_DNN_LIBRARY_LOG(WARNING)
          << "There is no LITTLE cores, so switch LOW to HIGH mode.";
      if (thread_num > big_num) {
        active_idxs_ = big_idxs;
        ARM_DNN_LIBRARY_LOG(WARNING)
            << "Apply for " << thread_num
            << " threads with HIGH mode, but exceed the number "
               "of the big cores "
            << big_num << ", so shrink to " << big_num << " threads.";
      } else {
        for (size_t i = 0; i < thread_num; i++) {
          active_idxs_.push_back(big_idxs[i]);
        }
      }
    }
  } else if (power_mode_ == PowerMode::FULL ||
             power_mode_ == PowerMode::NO_BIND) {
    for (size_t i = 0; i < thread_num; i++) {
      if (i < big_num) {
        active_idxs_.push_back(big_idxs[i]);
      } else {
        active_idxs_.push_back(little_idxs[i - big_num]);
      }
    }
  } else {
    ARM_DNN_LIBRARY_LOG(FATAL) << "Unsupported power mode "
                               << static_cast<int>(power_mode_) << ".";
  }
  if (power_mode_ == PowerMode::NO_BIND || bind_threads_to_cpus(active_idxs_)) {
    reset_to_no_bind = false;
  }
#endif
  if (reset_to_no_bind) {
    active_idxs_.clear();
    power_mode_ = PowerMode::NO_BIND;
    for (size_t i = 0; i < thread_num; i++) {
      if (i < big_num) {
        active_idxs_.push_back(big_idxs[i]);
      } else {
        active_idxs_.push_back(little_idxs[i - big_num]);
      }
    }
    ARM_DNN_LIBRARY_LOG(WARNING) << "Force to NO_BIND mode, use "
                                 << active_idxs_.size() << " threads.";
  }
  ARM_DNN_LIBRARY_CHECK_GE(active_idxs_.size(), 1);
#ifdef ARM_DNN_LIBRARY_WITH_OMP
  omp_set_num_threads(active_idxs_.size());
#endif
  return true;
}

void CPUInfo::DumpAllInfo() {
  auto cpu_num = cpu_attrs_.size();
  ARM_DNN_LIBRARY_LOG(INFO) << "Found: " << cpu_num << " CPUs.";
  for (size_t i = 0; i < cpu_num; i++) {
    const auto& cpu_attr = cpu_attrs_[i];
    ARM_DNN_LIBRARY_LOG(INFO) << "CPU[" << i << "]";
    ARM_DNN_LIBRARY_LOG(INFO) << " Arch: " << cpu_arch_to_string(cpu_attr.arch);
    ARM_DNN_LIBRARY_LOG(INFO)
        << " Cluster Id: "
        << (cpu_attr.cluster_id == 0
                ? "LITTLE"
                : (cpu_attr.cluster_id == 2 ? "big" : "Middle"));
    ARM_DNN_LIBRARY_LOG(INFO)
        << " L1 cache: " << cpu_attr.l1_cache_size / 1024.0f << " KB";
    ARM_DNN_LIBRARY_LOG(INFO)
        << " L2 cache: " << cpu_attr.l2_cache_size / 1024.0f << " KB";
    ARM_DNN_LIBRARY_LOG(INFO)
        << " L3 cache: " << cpu_attr.l3_cache_size / 1024.0f << " KB";
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm fp16: " << cpu_attr.support_arm_fp16;
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm bf16: " << cpu_attr.support_arm_bf16;
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm dotprod: "
                              << cpu_attr.support_arm_dotprod;
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm sve2: " << cpu_attr.support_arm_sve2;
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm sve2+i8mm: "
                              << cpu_attr.support_arm_sve2_i8mm;
    ARM_DNN_LIBRARY_LOG(INFO) << " Arm sve2+f32mm: "
                              << cpu_attr.support_arm_sve2_f32mm;
  }
}

}  // namespace armdnnlibrary
