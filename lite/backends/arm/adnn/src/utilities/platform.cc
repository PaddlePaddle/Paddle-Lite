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

#include "utilities/platform.h"
#include <vector>
#include "utilities/logging.h"

#if ADNN_OS_LINUX
#include <sys/syscall.h>
#include <unistd.h>
// http://elixir.free-electrons.com/linux/latest/source/arch/arm64/include/uapi/asm/hwcap.h
#include <asm/hwcap.h> /* Get HWCAP bits from asm/hwcap.h */
#include <sys/auxv.h>
#if ADNN_OS_ANDROID
#include <sys/system_properties.h>
#endif  // ADNN_OS_ANDROID
#elif ADNN_OS_IOS || ADNN_OS_MAC
#include "TargetConditionals.h"
#if ADNN_OS_IOS
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // ADNN_OS_IOS
#elif ADNN_OS_QNX
#include <sys/neutrino.h>
#endif  // ADNN_OS_LINUX

#ifdef ADNN_WITH_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <limits>
#include <string>

namespace adnn {

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
#endif

int get_cpu_num() {
  int cpu_num = 0;
#if ADNN_OS_LINUX
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
#elif ADNN_OS_IOS || ADNN_OS_MAC
  size_t size = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &size, NULL, 0);
#else   // ADNN_OS_LINUX
  ADNN_LOG("get_cpu_num() is not implemented, set to default 1.\n");
#endif  // ADNN_OS_LINUX
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
}

std::string get_cpu_name() {
  std::string cpu_name = "";
#if ADNN_OS_LINUX
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
#if ADNN_OS_ANDROID
  // cpu name concat board name, platform name and chip name
  char board_name[128];
  char platform_name[128];
  char chip_name[128];
  __system_property_get("ro.product.board", board_name);
  __system_property_get("ro.board.platform", platform_name);
  __system_property_get("ro.chipname", chip_name);
  cpu_name =
      cpu_name + "_" + board_name + "_" + platform_name + "_" + chip_name;
#endif  // ADNN_OS_ANDROID
  std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
  fclose(fp);
#endif  // ADNN_OS_LINUX
  return cpu_name;
}

size_t get_mem_size() {
  size_t mem_size = 0;
#if ADNN_OS_LINUX
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/meminfo", "rb");
  if (fp) {
    char line[1024];
    while (!feof(fp)) {
      char* s = fgets(line, 1024, fp);
      if (!s) {
        break;
      }
      sscanf(s, "MemTotal:        %d kB", &mem_size);
    }
    fclose(fp);
  }
#elif ADNN_OS_IOS || ADNN_OS_MAC
  mem_size = 4096 * 1024;
  ADNN_LOG(WARNING)
      << "get_mem_size() is not implemented, set to default 4GB.\n";
#else   // ADNN_OS_LINUX
  ADNN_LOG(WARNING) << "get_mem_size() is not implemented, set to default 0.\n";
#endif  // ADNN_OS_LINUX
  return mem_size;
}

void get_cpu_arch(std::vector<CPUArch>* cpu_archs, const int cpu_num) {
  cpu_archs->resize(cpu_num);
  for (int i = 0; i < cpu_num; ++i) {
    cpu_archs->at(i) = CPUArch::UNKOWN;
  }
#if ADNN_OS_LINUX
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return;
  }
  int cpu_idx = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "part") != NULL) {
      CPUArch arch_type = CPUArch::UNKOWN;
      int arch_id = 0;
      sscanf(s, "CPU part\t: %x", &arch_id);
      switch (arch_id) {
        case 0xd03:
          arch_type = CPUArch::CORTEX_A53;
          break;
        case 0xd04:
          arch_type = CPUArch::CORTEX_A35;
          break;
        case 0x803:
        case 0x805:
        case 0xd05:
          arch_type = CPUArch::CORTEX_A55;
          break;
        case 0xd07:
          arch_type = CPUArch::CORTEX_A57;
          break;
        case 0xd08:
        case 0x205:
          arch_type = CPUArch::CORTEX_A72;
          break;
        case 0x800:
        case 0x801:
        case 0xd09:
          arch_type = CPUArch::CORTEX_A73;
          break;
        case 0x802:
        case 0xd0a:
          arch_type = CPUArch::CORTEX_A75;
          break;
        case 0x804:
        case 0xd40:
          arch_type = CPUArch::CORTEX_A76;
          break;
        case 0xd0d:
          arch_type = CPUArch::CORTEX_A77;
          break;
        case 0xd41:
          // 888
          arch_type = CPUArch::CORTEX_A78;
          break;
        case 0xd44:
          // 888
          arch_type = CPUArch::CORTEX_X1;
          break;
        case 0xd46:
          arch_type = CPUArch::CORTEX_A510;
          break;
        case 0xd47:
          arch_type = CPUArch::CORTEX_A710;
          break;
        case 0xd48:
          arch_type = CPUArch::CORTEX_X2;
          break;
        default:
          ADNN_LOG(WARNING) << "Unknow cpu arch: " << arch_id;
      }
      cpu_archs->at(cpu_idx) = arch_type;
      cpu_idx++;
    }
  }
  fclose(fp);
  for (; cpu_idx > 0 && cpu_idx < cpu_num; ++cpu_idx) {
    cpu_archs->at(cpu_idx) = cpu_archs->at(cpu_idx - 1);
  }
#elif ADNN_OS_IOS
  for (int i = 0; i < cpu_num; ++i) {
    cpu_archs->at(i) = CPUArch::APPLE;
  }
#elif ADNN_OS_MAC && ADNN_ARCH_ARM64
  for (int i = 0; i < cpu_num; ++i) {
    cpu_archs->at(i) = CPUArch::CORTEX_X1;
  }
#endif  // ADNN_OS_LINUX
}

int get_min_freq_khz(int cpu_id) {
  int min_freq_khz = 0;
#if ADNN_OS_LINUX
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_min_freq",
           cpu_id);
  FILE* fp = fopen(path, "rb");
  if (fp) {
    fscanf(fp, "%d", &min_freq_khz);
    fclose(fp);
  }
  ADNN_CHECK_GT(min_freq_khz, 0)
      << "cpuinfo_min_freq should be greater than 0, but receive "
      << min_freq_khz;
#else   // ADNN_OS_LINUX
  ADNN_LOG("get_min_freq_khz() is not implemented, set to default 0.\n");
#endif  // ADNN_OS_LINUX
  return min_freq_khz;
}

int get_max_freq_khz(int cpu_id) {
  int max_freq_khz = 0;
#if ADNN_OS_LINUX
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
           cpu_id);
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpu_id);
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
             cpu_id);
    fp = fopen(path, "rb");
    if (fp) {
      fscanf(fp, "%d", &max_freq_khz);
      fclose(fp);
    }
  }
  ADNN_CHECK_GT(max_freq_khz, 0)
      << "cpuinfo_max_freq should be greater than 0, but receive "
      << max_freq_khz;
#else   // ADNN_OS_LINUX
  ADNN_LOG("get_max_freq_khz() is not implemented, set to default 0.\n");
#endif  // ADNN_OS_LINUX
  return max_freq_khz;
}

void sort_cpuid_by_max_freq(const std::vector<int>& max_freqs,
                            std::vector<int>* cpu_ids,
                            std::vector<int>* cluster_ids) {
  int cpu_num = max_freqs.size();
  if (cpu_num == 0) {
    return;
  }
  cpu_ids->resize(cpu_num);
  cluster_ids->resize(cpu_num);
  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
  }
  // Sort cpuid as big core first
  for (int i = 0; i < cpu_num; i++) {
    for (int j = i + 1; j < cpu_num; j++) {
      if (max_freqs[i] < max_freqs[j]) {
        // swap
        int tmp = cpu_ids->at(i);
        cpu_ids->at(i) = cpu_ids->at(j);
        cpu_ids->at(j) = tmp;
      }
    }
  }
  // SMP
  int mid_max_freq =
      (max_freqs[cpu_ids->at(0)] + max_freqs[cpu_ids->at(cpu_num - 1)]) / 2;
  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
    if (max_freqs[i] >= mid_max_freq) {
      cluster_ids->at(i) = 0;
    } else {
      cluster_ids->at(i) = 1;
    }
  }
}

void get_cpu_cache_size(int cpu_id,
                        int* l1_cache_size,
                        int* l2_cache_size,
                        int* l3_cache_size) {
  *l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  *l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  *l3_cache_size = DEFAULT_L3_CACHE_SIZE;
#if ADNN_OS_LINUX
  int max_cache_idx_num = 10;
  for (int i = 0; i < max_cache_idx_num; i++) {
    char path[256];
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level",
             cpu_id,
             i);
    FILE* fp = fopen(path, "rb");
    if (fp) {
      int level = -1;
      fscanf(fp, "%d", &level);
      fclose(fp);
      snprintf(path,
               sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cache/index%d/size",
               cpu_id,
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
#else   // ADNN_OS_LINUX
  ADNN_LOG(
      "get_cpu_cache_size() is not implemented, set to DEFAULT_L1_CACHE_SIZE, "
      "DEFAULT_L2_CACHE_SIZE and DEFAULT_L3_CACHE_SIZE.\n");
#endif  // ADNN_OS_LINUX
}

bool check_cpu_online(const std::vector<int>& cpu_ids) {
  if (cpu_ids.size() == 0) {
    return false;
  }
#if ADNN_OS_LINUX
  bool online = true;
  char path[256];
  for (int i = 0; i < cpu_ids.size(); ++i) {
    snprintf(
        path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", cpu_ids[i]);
    FILE* fp = fopen(path, "rb");
    int flag = 0;
    if (fp) {
      fscanf(fp, "%d", &flag);
      fclose(fp);
    } else {
      ADNN_LOG(WARNING) << "Failed to query the online statue of CPU id:"
                        << cpu_ids[i];
    }
    if (flag == 0) {
      online = false;
      ADNN_LOG(WARNING) << "CPU id:" << cpu_ids[i] << " is offine.";
    }
  }
  return online;
#else   // ADNN_OS_LINUX
  ADNN_LOG("check_cpu_online() is not implemented, set to offline.\n");
  return false;
#endif  // ADNN_OS_LINUX
}

bool set_sched_affinity(const std::vector<int>& cpu_ids) {
#define PD_CPU_SETSIZE 1024
#define PD__NCPUBITS (8 * sizeof(unsigned long))  // NOLINT
  typedef struct {
    unsigned long __bits[PD_CPU_SETSIZE / PD__NCPUBITS];  // NOLINT
  } cpu_set_t;

#define PD_CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu) / PD__NCPUBITS] |= (1UL << ((cpu) % PD__NCPUBITS)))

#define PD_CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

#if ADNN_OS_LINUX || ADNN_OS_QNX
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif  // __GLIBC__
  cpu_set_t mask;
  PD_CPU_ZERO(&mask);
  unsigned int RUNMASK = 0;
  for (int i = 0; i < cpu_ids.size(); ++i) {
#if ADNN_OS_QNX
    RMSK_SET(cpu_ids[i], &RUNMASK);
#else   // ADNN_OS_QNX
    PD_CPU_SET(cpu_ids[i], &mask);
#endif  // ADNN_OS_QNX
  }
#if ADNN_OS_QNX
  int syscallret = ThreadCtl(_NTO_TCTL_RUNMASK, (unsigned int*)RUNMASK);
#else   // ADNN_OS_QNX
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
#endif  // ADNN_OS_QNX
  return syscallret != 0;
#else   // ADNN_OS_LINUX || ADNN_OS_QNX
  ADNN_LOG("set_sched_affinity() is not implemented, set to offline.\n");
  return false;
#endif  // ADNN_OS_LINUX || ADNN_OS_QNX
}

bool bind_threads(const std::vector<int> cpu_ids) {
#ifdef ADNN_WITH_OMP
  int thread_num = cpu_ids.size();
  omp_set_num_threads(thread_num);
  std::vector<int> ssarets(thread_num, 0);
#pragma omp parallel for
  for (int i = 0; i < thread_num; i++) {
    ssarets[i] = set_sched_affinity(cpu_ids);
  }
  for (int i = 0; i < thread_num; i++) {
    if (ssarets[i] != 0) {
      ADNN_LOG(WARNING) << "Set cpu affinity failed, core id: " << cpu_ids[i];
      return false;
    }
  }
#else   // ADNN_WITH_OMP
  std::vector<int> first_cpu_id;
  first_cpu_id.push_back(cpu_ids[0]);
  int ssaret = set_sched_affinity(first_cpu_id);
  if (ssaret != 0) {
    ADNN_LOG(WARNING) << "Set cpu affinity failed, core id: " << cpu_ids[0];
    return false;
  }
#endif  // ADNN_WITH_OMP
  return true;
}

bool support_arm_sve2() {
#if ADNN_OS_ANDROID && ADNN_ARCH_ARM64
#define AT_HWCAP2 26
#define AARCH64_HWCAP2_SVE2 (1UL << 1)
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVE2) return true;
#undef AARCH64_HWCAP2_SVE2
#undef AT_HWCAP2
#endif
  return false;
}

bool support_arm_sve2_i8mm() {
#if ADNN_OS_ANDROID && ADNN_ARCH_ARM64
#define AT_HWCAP2 26
#define AARCH64_HWCAP2_SVEI8MM (1UL << 9)
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVEI8MM) return true;
#undef AARCH64_HWCAP2_SVEI8MM
#undef AT_HWCAP2
#endif
  return false;
}

bool support_arm_sve2_f32mm() {
#if ADNN_OS_ANDROID && ADNN_ARCH_ARM64
#define AT_HWCAP2 26
#define AARCH64_HWCAP2_SVEF32MM (1UL << 10)
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVEF32MM) return true;
#undef AARCH64_HWCAP2_SVEF32MM
#undef AT_HWCAP2
#endif
  return false;
}

}  // namespace adnn
