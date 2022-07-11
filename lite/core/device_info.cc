// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef LITE_WITH_LINUX

#ifdef LITE_WITH_QNX
#include <sys/neutrino.h>
#else
#include <sys/syscall.h>
#endif

#include <unistd.h>
#endif
#ifdef LITE_WITH_ANDROID
#include <sys/system_properties.h>
#endif
#if __APPLE__
#include "TargetConditionals.h"
#if LITE_WITH_IPHONE
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // LITE_WITH_IPHONE
#endif  // __APPLE__

#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

#ifdef LITE_WITH_X86
#ifdef _WIN32
// For __cpuid
#include <intrin.h>
#endif
#endif

#include <algorithm>
#include <limits>
#include "lite/core/device_info.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
// http://elixir.free-electrons.com/linux/latest/source/arch/arm64/include/uapi/asm/hwcap.h
#if defined(LITE_WITH_ANDROID) && defined(__aarch64__)
#include <asm/hwcap.h> /* Get HWCAP bits from asm/hwcap.h */
#include <sys/auxv.h>
#define AARCH64_HWCAP_SVE (1UL << 22)
#define AARCH64_HWCAP2_SVE2 (1UL << 1)
#define AARCH64_HWCAP2_SVEAES (1UL << 2)
#define AARCH64_HWCAP2_SVEPMULL (1UL << 3)
#define AARCH64_HWCAP2_SVEBITPERM (1UL << 4)
#define AARCH64_HWCAP2_SVESHA3 (1UL << 5)
#define AARCH64_HWCAP2_SVESM4 (1UL << 6)
#define AARCH64_HWCAP2_SVEI8MM (1UL << 9)
#define AARCH64_HWCAP2_SVEF32MM (1UL << 10)
#define AARCH64_HWCAP2_SVEF64MM (1UL << 11)
#define AARCH64_HWCAP2_SVEBF16 (1UL << 12)
#define AARCH64_HWCAP2_I8MM (1UL << 13)
#define AARCH64_HWCAP2_BF16 (1UL << 14)
#define AT_HWCAP 16
#define AT_HWCAP2 26

bool check_sve2_valid() {
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVE2) return true;
  return false;
}

bool check_sve2_i8mm_vaild() {
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVEI8MM) return true;
  return false;
}

bool check_sve2_f32mm_vaild() {
  auto mask = static_cast<uint32_t>(getauxval(AT_HWCAP2));  // Android API >= 18
  if (mask & AARCH64_HWCAP2_SVEF32MM) return true;
  return false;
}
#endif

#if ((defined LITE_WITH_ARM) || (defined LITE_WITH_MLU))
LITE_THREAD_LOCAL lite_api::PowerMode DeviceInfo::mode_;
LITE_THREAD_LOCAL ARMArch DeviceInfo::arch_;
LITE_THREAD_LOCAL int DeviceInfo::mem_size_;
LITE_THREAD_LOCAL std::vector<int> DeviceInfo::active_ids_;
LITE_THREAD_LOCAL TensorLite DeviceInfo::workspace_;
LITE_THREAD_LOCAL int64_t DeviceInfo::count_ = 0;

#ifdef TARGET_IOS
const int DEFAULT_L1_CACHE_SIZE = 64 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#elif defined(LITE_WITH_M1)
const int DEFAULT_L1_CACHE_SIZE = 128 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 4096 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#else
const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 512 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#endif

int get_cpu_num() {
#ifdef LITE_WITH_LINUX
  // get cpu count from /sys/devices/system/cpu/cpunum/uevent
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
#elif defined(TARGET_IOS) || defined(LITE_WITH_M1)
  int cpu_num = 0;
  size_t len = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &len, NULL, 0);
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#else
  return 1;
#endif
}

size_t get_mem_size() {
#ifdef LITE_WITH_LINUX
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/meminfo", "rb");
  if (!fp) {
    return 1;
  }
  size_t memsize = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    sscanf(s, "MemTotal:        %d kB", &memsize);
  }
  fclose(fp);
  return memsize;
#elif defined(TARGET_IOS) || defined(LITE_WITH_M1)
  // to be implemented
  printf("not implemented, set to default 4GB\n");
  return 4096 * 1024;
#endif
  return 0;
}

void get_cpu_arch(std::vector<ARMArch>* archs, const int cpu_num) {
  archs->resize(cpu_num);
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kARMArch_UNKOWN;
  }
#ifdef LITE_WITH_LINUX
  //! get CPU ARCH
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
      ARMArch arch_type = kARMArch_UNKOWN;
      int arch_id = 0;
      sscanf(s, "CPU part\t: %x", &arch_id);
      switch (arch_id) {
        case 0xd03:
          arch_type = kA53;
          break;
        case 0xd04:
          arch_type = kA35;
          break;
        case 0x803:
        case 0x805:
        case 0xd05:
          arch_type = kA55;
          break;
        case 0xd07:
          arch_type = kA57;
          break;
        case 0xd08:
        case 0x205:
          arch_type = kA72;
          break;
        case 0x800:
        case 0x801:
        case 0xd09:
          arch_type = kA73;
          break;
        case 0x802:
        case 0xd0a:
          arch_type = kA75;
          break;
        case 0x804:
        case 0xd40:
          arch_type = kA76;
          break;
        case 0xd0d:
          arch_type = kA77;
          break;
        case 0xd41:
          // 888
          arch_type = kA78;
          break;
        case 0xd44:
          // 888
          arch_type = kX1;
          break;
        case 0xd46:
          arch_type = kA510;
          break;
        case 0xd47:
          arch_type = kA710;
          break;
        case 0xd48:
          arch_type = kX2;
          break;
        default:
          LOG(ERROR) << "Unknow cpu arch: " << arch_id;
      }
      archs->at(cpu_idx) = arch_type;
      cpu_idx++;
    }
  }
  fclose(fp);
  for (; cpu_idx > 0 && cpu_idx < cpu_num; ++cpu_idx) {
    archs->at(cpu_idx) = archs->at(cpu_idx - 1);
  }
#elif defined(TARGET_IOS)
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kAPPLE;
  }
#elif defined(LITE_WITH_M1)
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kX1;
  }
#endif
}

#ifdef LITE_WITH_LINUX

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
#ifdef LITE_WITH_ANDROID
  // cpu name concat board name, platform name and chip name
  char board_name[128];
  char platform_name[128];
  char chip_name[128];
  __system_property_get("ro.product.board", board_name);
  __system_property_get("ro.board.platform", platform_name);
  __system_property_get("ro.chipname", chip_name);
  cpu_name =
      cpu_name + "_" + board_name + "_" + platform_name + "_" + chip_name;
#endif
  std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
  fclose(fp);
  return cpu_name;
}

int get_min_freq_khz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
           cpuid);
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    return -1;
  }

  int min_freq_khz = -1;
  fscanf(fp, "%d", &min_freq_khz);
  fclose(fp);
  return min_freq_khz;
}

int get_max_freq_khz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
           cpuid);

  FILE* fp = fopen(path, "rb");
  if (!fp) {
    // second try, for online cpu
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpuid);
    fp = fopen(path, "rb");
  }

  int max_freq_khz = 0;
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
  }
  if (max_freq_khz == 0 || !fp) {
    // third try, for online cpu
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
             cpuid);
    fp = fopen(path, "rb");
    if (!fp) {
      return -1;
    }
    int max_freq_khz = -1;
    fscanf(fp, "%d", &max_freq_khz);
    fclose(fp);
    return max_freq_khz;
  }

  fclose(fp);
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
  // sort cpuid as big core first
  // simple bubble sort
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
  int max_cache_idx_num = 10;
  *l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  *l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  *l3_cache_size = DEFAULT_L3_CACHE_SIZE;
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
}

bool check_cpu_online(const std::vector<int>& cpu_ids) {
  if (cpu_ids.size() == 0) {
    return false;
  }
  char path[256];
  bool all_online = true;
  for (int i = 0; i < cpu_ids.size(); ++i) {
    snprintf(
        path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", cpu_ids[i]);
    FILE* fp = fopen(path, "rb");
    int is_online = 0;
    if (fp) {
      fscanf(fp, "%d", &is_online);
      fclose(fp);
    } else {
      LOG(ERROR) << "Failed to query the online statue of CPU id:"
                 << cpu_ids[i];
    }
    if (is_online == 0) {
      all_online = false;
      LOG(ERROR) << "CPU id:" << cpu_ids[i] << " is offine";
    }
  }
  return all_online;
}

int set_sched_affinity(const std::vector<int>& cpu_ids) {
#define PD_CPU_SETSIZE 1024
#define PD__NCPUBITS (8 * sizeof(unsigned long))  // NOLINT
  typedef struct {
    unsigned long __bits[PD_CPU_SETSIZE / PD__NCPUBITS];  // NOLINT
  } cpu_set_t;

#define PD_CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu) / PD__NCPUBITS] |= (1UL << ((cpu) % PD__NCPUBITS)))

#define PD_CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

// set affinity for thread
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif
  cpu_set_t mask;
  PD_CPU_ZERO(&mask);
  unsigned int Runmask = 0;
  for (int i = 0; i < cpu_ids.size(); ++i) {
#ifdef LITE_WITH_QNX
    RMSK_SET(cpu_ids[i], &Runmask);  // set CPU
#else
    PD_CPU_SET(cpu_ids[i], &mask);
#endif
  }
#ifdef LITE_WITH_QNX
  int syscallret = ThreadCtl(_NTO_TCTL_RUNMASK, (unsigned int*)Runmask);
#else
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
#endif
  if (syscallret) {
    return -1;
  }
  return 0;
}

bool bind_threads(const std::vector<int> cpu_ids) {
#ifdef ARM_WITH_OMP
  int thread_num = cpu_ids.size();
  omp_set_num_threads(thread_num);
  std::vector<int> ssarets(thread_num, 0);
#pragma omp parallel for
  for (int i = 0; i < thread_num; i++) {
    ssarets[i] = set_sched_affinity(cpu_ids);
  }
  for (int i = 0; i < thread_num; i++) {
    if (ssarets[i] != 0) {
      LOG(ERROR) << "Set cpu affinity failed, core id: " << cpu_ids[i];
      return false;
    }
  }
#else   // ARM_WITH_OMP
  std::vector<int> first_cpu_id;
  first_cpu_id.push_back(cpu_ids[0]);
  int ssaret = set_sched_affinity(first_cpu_id);
  if (ssaret != 0) {
    LOG(ERROR) << "Set cpu affinity failed, core id: " << cpu_ids[0];
    return false;
  }
#endif  // ARM_WITH_OMP
  return true;
}

#endif  // LITE_WITH_LINUX

void DeviceInfo::SetDotInfo(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  dot_.resize(core_num_);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (int i = 0; i < core_num_; ++i) {
      dot_[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      dot_[big_core_ids_[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      dot_[little_core_ids_[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

void DeviceInfo::SetFP16Info(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  fp16_.resize(core_num_);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (int i = 0; i < core_num_; ++i) {
      fp16_[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      fp16_[big_core_ids_[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      fp16_[little_core_ids_[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

void DeviceInfo::SetFP32Info(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  fp32_.resize(core_num_);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (int i = 0; i < core_num_; ++i) {
      fp32_[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      fp32_[big_core_ids_[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      fp32_[little_core_ids_[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

// cache_id : 0 -> L1, 1 -> L2, 2 -> L3
void DeviceInfo::SetCacheInfo(int cache_id, int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  std::vector<int>* cache = nullptr;
  switch (cache_id) {
    case 0:
      cache = &L1_cache_;
      break;
    case 1:
      cache = &L2_cache_;
      break;
    case 2:
      cache = &L3_cache_;
      break;
    default:
      break;
  }
  cache->resize(core_num_);
  if (argc == 1) {
    int cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < core_num_; ++i) {
      (*cache)[i] = cache_size;
    }
  } else if (argc == 2) {
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    int big_core_cache_size = va_arg(arg_ptr, int);
    int little_core_cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < big_core_num; ++i) {
      (*cache)[big_core_ids_[i]] = big_core_cache_size;
    }
    for (int i = 0; i < little_core_num; ++i) {
      (*cache)[little_core_ids_[i]] = little_core_cache_size;
    }
  } else if (argc == 3) {
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    int big_core_cache_size0 = va_arg(arg_ptr, int);
    int big_core_cache_size1 = va_arg(arg_ptr, int);
    int little_core_cache_size = va_arg(arg_ptr, int);
    (*cache)[big_core_ids_[big_core_num - 1]] = big_core_cache_size0;
    for (int i = 0; i < big_core_num - 1; ++i) {
      (*cache)[big_core_ids_[i]] = big_core_cache_size1;
    }
    for (int i = 0; i < little_core_num; ++i) {
      (*cache)[little_core_ids_[i]] = little_core_cache_size;
    }
  }
  va_end(arg_ptr);
}

void DeviceInfo::SetArchInfo(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  archs_.resize(core_num_);
  if (argc == 1) {
    ARMArch arch = (ARMArch)va_arg(arg_ptr, int);
    for (int i = 0; i < core_num_; ++i) {
      archs_[i] = arch;
    }
  } else if (argc == 2) {
    ARMArch big_core_arch = (ARMArch)va_arg(arg_ptr, int);
    ARMArch little_core_arch = (ARMArch)va_arg(arg_ptr, int);
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      archs_[big_core_ids_[i]] = big_core_arch;
    }
    for (int i = 0; i < little_core_num; ++i) {
      archs_[little_core_ids_[i]] = little_core_arch;
    }
  } else if (argc == 3) {
    // 888
    ARMArch big_core_arch0 = (ARMArch)va_arg(arg_ptr, int);
    ARMArch big_core_arch1 = (ARMArch)va_arg(arg_ptr, int);
    ARMArch little_core_arch = (ARMArch)va_arg(arg_ptr, int);
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    archs_[big_core_ids_[big_core_num - 1]] = big_core_arch0;
    for (int i = 0; i < big_core_num - 1; ++i) {
      archs_[big_core_ids_[i]] = big_core_arch1;
    }
    for (int i = 0; i < little_core_num; ++i) {
      archs_[little_core_ids_[i]] = little_core_arch;
    }
  }
  va_end(arg_ptr);
}

bool DeviceInfo::SetCPUInfoByName() {
  /* Snapdragon */
  if (dev_name_.find("SM8350") != std::string::npos) {  // 888
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(3, kX1, kA78, kA55);
    SetCacheInfo(0, 3, 512 * 1024, 192 * 1024, 256 * 1024);
    SetCacheInfo(1, 3, 1024 * 1024, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4 * 1024 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(2, 1, 1);
    return true;
  } else if (dev_name_.find("SA8155") != std::string::npos) {  // sa8155
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(3, kGold_Prime, kGold, kSilver);
    SetCacheInfo(0, 3, 512 * 1024, 256 * 1024, 128 * 1024);
    SetCacheInfo(1, 3, 512 * 1024, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2 * 1024 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(2, 1, 1);
    return true;
  } else if (dev_name_.find("SA8195") != std::string::npos) {  // sa8195
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kGold_Prime, kSilver);
    SetCacheInfo(0, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4 * 1024 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(2, 1, 1);
    return true;
  } else if (dev_name_.find("KONA") != std::string::npos) {  // 865
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA77, kA55);
    SetCacheInfo(0, 2, 192 * 1024, 256 * 1024);
    SetCacheInfo(1, 2, 768 * 1024, 512 * 1024);
    SetCacheInfo(2, 1, 4 * 1024 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(2, 1, 1);
    return true;
  } else if (dev_name_.find("SM8150") != std::string::npos) {  // 855
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
  } else if (dev_name_.find("SDM845") != std::string::npos) {  // 845
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    SetFP16Info(1, 1);
    return true;
  } else if (dev_name_.find("SDM710") != std::string::npos) {  // 710
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {6, 7};
    little_core_ids_ = {0, 1, 2, 3, 4, 5};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8998") != std::string::npos) {  // 835
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1,
                 2,
                 1024 * 1024,
                 /*real cache size is 2M, while that will get bad performace
                    on conv3x3s1 or gemm, set to 1M or 512K*/
                 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8996") != std::string::npos) {  // 820
    core_num_ = 4;
    core_ids_ = {0, 1, 2, 3};
    big_core_ids_ = {2, 3};
    little_core_ids_ = {0, 1};
    cluster_ids_ = {1, 1, 0, 0};
    SetArchInfo(1, kA72);
    SetCacheInfo(0, 1, 24 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("SDM660") != std::string::npos ||
             dev_name_.find("SDM636") != std::string::npos) {  // 660, 636
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA73);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8976") != std::string::npos) {  // 652,653
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("MSM8953") != std::string::npos) {  // 625
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8939") != std::string::npos) {  // 615
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3};
    little_core_ids_ = {4, 5, 6, 7};
    cluster_ids_ = {0, 0, 0, 0, 1, 1, 1, 1};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 256 * 1024);
    return true;
    /* MediaTek */
  } else if (dev_name_.find("MT6891") != std::string::npos) {  // Dimensity 1100
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA78, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 64 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4 * 1024 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(2, 1, 1);
    return true;
  } else if (dev_name_.find("MT6797") !=
             std::string::npos) {  // X20/X23/X25/X27
    core_num_ = 10;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    big_core_ids_ = {8, 9};
    little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("MT6799") != std::string::npos) {  // X30
    core_num_ = 10;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    big_core_ids_ = {8, 9};
    little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (dev_name_.find("MT6795") != std::string::npos ||
             dev_name_.find("MT6762") != std::string::npos ||
             dev_name_.find("MT6755T") != std::string::npos ||
             dev_name_.find("MT6755S") != std::string::npos ||
             dev_name_.find("MT6753") != std::string::npos ||
             dev_name_.find("MT6752") != std::string::npos ||
             dev_name_.find("MT6750") != std::string::npos) {
    // X10, P22, P15/P18, MT6753, MT6752/MT6752M, MT6750
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (dev_name_.find("MT6758") != std::string::npos ||
             dev_name_.find("MT6757") != std::string::npos ||
             dev_name_.find("MT6763") != std::string::npos ||
             dev_name_.find("MT6755M") != std::string::npos ||
             dev_name_.find("MT6755") !=
                 std::string::npos) {  // P30, P20/P25, P23, P10
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (dev_name_.find("MT6771") != std::string::npos) {  // P60
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (dev_name_.find("MT6765") != std::string::npos ||
             dev_name_.find("MT6739") != std::string::npos ||
             dev_name_.find("MT6738") != std::string::npos ||
             dev_name_.find("MT6737") !=
                 std::string::npos) {  // A22, MT6739, MT6738, MT6767
    core_num_ = 4;
    core_ids_ = {0, 1, 2, 3};
    big_core_ids_ = {0, 1, 2, 3};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (dev_name_.find("KIRIN980") != std::string::npos ||
             dev_name_.find("KIRIN990") !=
                 std::string::npos) {  // Kirin 980, Kirin 990
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4096 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
  } else if (dev_name_.find("KIRIN810") != std::string::npos) {  // Kirin 810
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {6, 7};
    little_core_ids_ = {0, 1, 2, 3, 4, 5};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
  } else if (dev_name_.find("FT2000PLUS") != std::string::npos) {
    core_num_ = 64;
    core_ids_.resize(core_num_);
    big_core_ids_.resize(core_num_);
    cluster_ids_.resize(core_num_);
    for (int i = 0; i < core_num_; ++i) {
      core_ids_[i] = i;
      big_core_ids_[i] = i;
      cluster_ids_[i] = 0;
    }
    little_core_ids_ = {};
    SetCacheInfo(0, 1, 64 * 1024);
    SetCacheInfo(1, 1, 32 * 1024 * 1024);
    SetCacheInfo(2, 1, 128 * 1024 * 1024);
    return true;
  }
  return false;
}

void DeviceInfo::SetCPUInfoByProb() {
#ifdef LITE_WITH_LINUX
  // get big.LITTLE cores by sorting CPU frequency
  sort_cpuid_by_max_freq(max_freqs_, &core_ids_, &cluster_ids_);
  big_core_ids_.clear();
  little_core_ids_.clear();
  for (int i = 0; i < cluster_ids_.size(); ++i) {
    if (cluster_ids_[i] == 0) {
      big_core_ids_.push_back(core_ids_[i]);
    } else {
      little_core_ids_.push_back(core_ids_[i]);
    }
  }
  // get l1, l2, l3 cache size for each core
  for (int i = 0; i < core_num_; i++) {
    get_cpu_cache_size(i, &(L1_cache_[i]), &(L2_cache_[i]), &(L3_cache_[i]));
  }
#endif  // LITE_WITH_LINUX
}

void DeviceInfo::RequestPowerFullMode(int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  for (int i = 0; i < thread_num; ++i) {
    if (i < big_core_size) {
      active_ids_.push_back(big_core_ids_[i]);
    } else if (i < big_core_size + little_core_size) {
      active_ids_.push_back(little_core_ids_[i - big_core_size]);
    }
  }
  mode_ = lite_api::PowerMode::LITE_POWER_FULL;
}

void DeviceInfo::RequestPowerHighMode(int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (big_core_size > 0) {
    mode_ = lite_api::PowerMode::LITE_POWER_HIGH;
    if (thread_num > big_core_size) {
      LOG(ERROR) << "Request thread num: " << thread_num
                 << ", exceed the big cores size: " << big_core_size
                 << ", truncate thread num to " << big_core_size;
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[big_core_size - 1 - i]);
      }
    }
  } else {
    mode_ = lite_api::PowerMode::LITE_POWER_LOW;
    LOG(ERROR) << "HIGH POWER MODE is not support, switch to little cores.";
    if (thread_num > little_core_size) {
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerLowMode(int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (little_core_size > 0) {
    mode_ = lite_api::PowerMode::LITE_POWER_LOW;
    if (thread_num > little_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the little cores size: " << little_core_size
                   << ", truncate thread num to " << little_core_size;
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; i++) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  } else {
    mode_ = lite_api::PowerMode::LITE_POWER_HIGH;
    LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores";
    if (thread_num > big_core_size) {
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; i++) {
        active_ids_.push_back(big_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerNoBindMode(int thread_num) {
  active_ids_.clear();
  if (thread_num > core_ids_.size()) {
    active_ids_ = core_ids_;
  } else {
    active_ids_.resize(thread_num);
    for (uint32_t i = 0; i < thread_num; ++i) {
      if (i < big_core_ids_.size()) {
        active_ids_[i] = big_core_ids_[i];
      } else {
        active_ids_[i] = little_core_ids_[i - big_core_ids_.size()];
      }
    }
  }
  mode_ = lite_api::PowerMode::LITE_POWER_NO_BIND;
}

void DeviceInfo::RequestPowerRandHighMode(int shift_num, int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (big_core_size > 0) {
    mode_ = lite_api::PowerMode::LITE_POWER_RAND_HIGH;
    if (thread_num > big_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the big cores size: " << big_core_size
                   << ", truncate thread num to " << big_core_size;
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[(i + shift_num) % big_core_size]);
      }
    }
  } else {
    mode_ = lite_api::PowerMode::LITE_POWER_LOW;
    LOG(WARNING) << "HIGH POWER MODE is not support, switch to little cores.";
    if (thread_num > little_core_size) {
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerRandLowMode(int shift_num, int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (little_core_size > 0) {
    mode_ = lite_api::PowerMode::LITE_POWER_RAND_LOW;
    if (thread_num > little_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the little cores size: " << little_core_size
                   << ", truncate thread num to " << little_core_size;
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(
            little_core_ids_[(i + shift_num) % little_core_size]);
      }
    }
  } else {
    mode_ = lite_api::PowerMode::LITE_POWER_HIGH;
    LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores.";
    if (thread_num > big_core_size) {
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[i]);
      }
    }
  }
}

bool DeviceInfo::set_a53_valid() { return has_a53_valid_; }

bool DeviceInfo::has_sve2() { return has_sve2_; }

bool DeviceInfo::has_sve2_f32mm() { return has_sve2_f32mm_; }

bool DeviceInfo::has_sve2_i8mm() { return has_sve2_i8mm_; }

int DeviceInfo::Setup() {
  core_num_ = get_cpu_num();
  mem_size_ = get_mem_size();
  get_cpu_arch(&archs_, core_num_);
  // set defalut CPU info
  SetCacheInfo(0, 1, DEFAULT_L1_CACHE_SIZE);
  SetCacheInfo(1, 1, DEFAULT_L2_CACHE_SIZE);
  SetCacheInfo(2, 1, DEFAULT_L3_CACHE_SIZE);
  SetFP32Info(1, 1);
  SetFP16Info(1, 0);
  SetDotInfo(1, 0);
  max_freqs_.resize(core_num_);
  min_freqs_.resize(core_num_);
#ifdef LITE_WITH_LINUX
  // get max&min freq
  for (int i = 0; i < core_num_; ++i) {
    int max_freq = get_max_freq_khz(i);
    int min_freq = get_min_freq_khz(i);
    max_freqs_[i] = max_freq / 1000;
    min_freqs_[i] = min_freq / 1000;
  }
  // get cache size and big.LITTLE core ids
  dev_name_ = get_cpu_name();
  if (!SetCPUInfoByName()) {
    SetCPUInfoByProb();
  }
#else
#ifdef TARGET_IOS
  dev_name_ = "Apple";
#elif defined(LITE_WITH_M1)
  dev_name_ = "M1";
  SetDotInfo(1, 1);
  SetFP16Info(1, 1);
#else
  dev_name_ = "Unknown";
#endif
  core_ids_.resize(core_num_);
  cluster_ids_.resize(core_num_);
  big_core_ids_.resize(core_num_);
  for (int i = 0; i < core_num_; ++i) {
    max_freqs_[i] = 1000000;
    min_freqs_[i] = 1000000;
    cluster_ids_[i] = 0;
    core_ids_[i] = i;
    big_core_ids_[i] = i;
  }
#endif
  // xiaodu device_name
  if (dev_name_.find("MT8765WA") != std::string::npos ||
      dev_name_.find("MT8167S") != std::string::npos) {
    has_a53_valid_ = false;
  } else {
    has_a53_valid_ = true;
  }

  // SVE2
  has_sve2_ = false;
  has_sve2_i8mm_ = false;
  has_sve2_f32mm_ = false;
#if defined(LITE_WITH_ANDROID) && defined(__aarch64__)
  has_sve2_ = check_sve2_valid();
  has_sve2_f32mm_ = has_sve2_ && check_sve2_f32mm_vaild();
  has_sve2_i8mm_ = has_sve2_ && check_sve2_i8mm_vaild();
#endif

  // output info
  LOG(INFO) << "ARM multiprocessors name: " << dev_name_;
  LOG(INFO) << "ARM multiprocessors number: " << core_num_;
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << "ARM multiprocessors ID: " << core_ids_[i]
              << ", max freq: " << max_freqs_[i]
              << ", min freq: " << min_freqs_[i]
              << ", cluster ID: " << cluster_ids_[core_ids_[i]]
              << ", CPU ARCH: A" << static_cast<int>(archs_[i]);
  }
  LOG(INFO) << "L1 DataCache size is: ";
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << L1_cache_[i] / 1024 << " KB";
  }
  LOG(INFO) << "L2 Cache size is: ";
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << L2_cache_[i] / 1024 << " KB";
  }
  LOG(INFO) << "L3 Cache size is: ";
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << L3_cache_[i] / 1024 << " KB";
  }
  LOG(INFO) << "Total memory: " << mem_size_ << "KB";
  LOG(INFO) << "SVE2 support: " << has_sve2_;
  LOG(INFO) << "SVE2 f32mm support: " << has_sve2_f32mm_;
  LOG(INFO) << "SVE2 i8mm support: " << has_sve2_i8mm_;
  // set default run mode
  SetRunMode(lite_api::PowerMode::LITE_POWER_NO_BIND,
             1);  // use single thread by default
  return 0;
}

void DeviceInfo::SetRunMode(lite_api::PowerMode mode, int thread_num) {
#if defined(ARM_WITH_OMP) || defined(LITE_USE_THREAD_POOL)
  thread_num = std::min(thread_num, core_num_);
#else
  thread_num = 1;  // force thread_num to 1 if OpenMP is disabled
#endif
#ifdef LITE_WITH_LINUX
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  int big_little_core_size = big_core_size + little_core_size;
  thread_num = std::min(thread_num, big_little_core_size);
  count_++;
  int shift_num = (count_ / 10) % big_core_size;
  switch (mode) {
    case lite_api::LITE_POWER_FULL:
      RequestPowerFullMode(thread_num);
      break;
    case lite_api::LITE_POWER_HIGH:
      RequestPowerHighMode(thread_num);
      break;
    case lite_api::LITE_POWER_LOW:
      RequestPowerLowMode(thread_num);
      break;
    case lite_api::LITE_POWER_NO_BIND:
      RequestPowerNoBindMode(thread_num);
      break;
    case lite_api::LITE_POWER_RAND_HIGH:
      RequestPowerRandHighMode(shift_num, thread_num);
      break;
    case lite_api::LITE_POWER_RAND_LOW:
      RequestPowerRandLowMode(shift_num, thread_num);
      break;
    default:
      LOG(FATAL) << "Unsupported power mode: " << static_cast<int>(mode);
      break;
  }
  if (active_ids_.empty()) {
    active_ids_.push_back(0);
  }
#ifdef ARM_WITH_OMP
  omp_set_num_threads(active_ids_.size());
#endif
  if (mode_ != lite_api::LITE_POWER_NO_BIND) {
    if (check_cpu_online(active_ids_)) {
      bind_threads(active_ids_);
    } else {
      LOG(WARNING) << "Some cores are offline, switch to NO BIND MODE";
      mode_ = lite_api::LITE_POWER_NO_BIND;
    }
  }
#else  // LITE_WITH_LINUX
  // only LITE_POWER_NO_BIND is supported in other OS
  RequestPowerNoBindMode(thread_num);
#ifdef ARM_WITH_OMP
  omp_set_num_threads(active_ids_.size());
#endif
#endif  // LITE_WITH_LINUX
  //! alloc memory for sgemm in this context
  workspace_.Resize({llc_size()});
  workspace_.mutable_data<int8_t>();
  arch_ = archs_[active_ids_[0]];
}

void DeviceInfo::SetCache(int l1size, int l2size, int l3size) {
  SetCacheInfo(0, 1, l1size);
  SetCacheInfo(1, 1, l2size);
  SetCacheInfo(2, 1, l3size);
  workspace_.Resize({llc_size()});
  workspace_.mutable_data<int8_t>();
}

bool DeviceInfo::ExtendWorkspace(size_t size) {
  workspace_.Resize(
      {static_cast<int64_t>(size + static_cast<size_t>(llc_size()))});
  return workspace_.mutable_data<int8_t>() != nullptr;
}

#endif  // LITE_WITH_ARM

#ifdef LITE_WITH_MLU
void SetMluDevice(int device_id) {
  LOG(INFO) << "Set mlu device " << device_id;
  cnrtDev_t dev_handle;
  CNRT_CALL(cnrtGetDeviceHandle(&dev_handle, device_id));
  CNRT_CALL(cnrtSetCurrentDevice(dev_handle));
}

void Device<TARGET(kMLU)>::Init() {
  SetMluDevice(idx_);
  GetInfo();
  CreateQueue();
}

void Device<TARGET(kMLU)>::GetInfo() {}

void Device<TARGET(kMLU)>::CreateQueue() {
  exec_queue_.clear();
  io_queue_.clear();
  for (size_t i = 0; i < max_queue_; ++i) {
    cnrtQueue_t exec_queue;
    cnrtQueue_t io_queue;
    cnrtCreateQueue(&exec_queue);
    cnrtCreateQueue(&io_queue);
    exec_queue_.push_back(exec_queue);
    io_queue_.push_back(io_queue);

    cnrtCreateQueue(&exec_queue);
    exec_queue_.push_back(exec_queue);
  }
}
#endif  // LITE_WITH_MLU

#ifdef LITE_WITH_BM
void Device<TARGET(kBM)>::SetId(int device_id) {
  LOG(INFO) << "Set bm device " << device_id;
  TargetWrapper<TARGET(kBM)>::SetDevice(device_id);
  idx_ = device_id;
}

void Device<TARGET(kBM)>::Init() { SetId(idx_); }
int Device<TARGET(kBM)>::core_num() {
  return TargetWrapper<TARGET(kBM)>::num_devices();
}
#endif  // LITE_WITH_BM

#ifdef LITE_WITH_CUDA

void Device<TARGET(kCUDA)>::Init() {
  GetInfo();
  CreateStream();
}

void Device<TARGET(kCUDA)>::GetInfo() {
  cudaGetDeviceProperties(&device_prop_, idx_);
  cudaRuntimeGetVersion(&runtime_version_);
  sm_version_ = (device_prop_.major << 8 | device_prop_.minor);
  has_hmma_ =
      (sm_version_ == 0x0700 || sm_version_ == 0x0702 || sm_version_ == 0x0705);
  has_fp16_ = (sm_version_ == 0x0602 || sm_version_ == 0x0600 ||
               sm_version_ == 0x0503 || has_hmma_);
  has_imma_ = (sm_version_ == 0x0702 || sm_version_ == 0x0705);
  has_int8_ = (sm_version_ == 0x0601 || sm_version_ == 0x0700 || has_imma_);
}

void Device<TARGET(kCUDA)>::CreateStream() {
  exec_stream_.clear();
  io_stream_.clear();
  for (int i = 0; i < max_stream_; i++) {
    cudaStream_t exec_stream;
    cudaStream_t io_stream;
    cudaStreamCreate(&exec_stream);
    cudaStreamCreate(&io_stream);
    exec_stream_.push_back(exec_stream);
    io_stream_.push_back(io_stream);
  }
}

#endif

#ifdef LITE_WITH_X86

#define uint32_t unsigned int

struct CPUID {
  uint32_t eax, ebx, ecx, edx;
  CPUID() {
#if defined(_WIN32)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    eax = cpuInfo[0];
    ebx = cpuInfo[1];
    ecx = cpuInfo[2];
    edx = cpuInfo[3];
#else
    asm volatile("cpuid\n"
                 : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                 : "a"(1)
                 : "cc");
#endif
  }
} cpuid;

bool bit(unsigned x, unsigned y) { return (x >> y) & 1; }

bool feature_detect_avx2() {
  uint32_t eax, ebx, ecx, edx;

// check cpu support
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  eax = cpuInfo[0];
  ebx = cpuInfo[1];
  ecx = cpuInfo[2];
  edx = cpuInfo[3];
#else
  asm volatile("cpuid\n"
               : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
               : "a"(7), "c"(0)
               : "cc");
#endif

  if (!(bit(ebx, 3) && bit(ebx, 5) && bit(ebx, 8))) return false;

// check os support ymm or xmm
#if defined(_WIN32)
  eax = _xgetbv(0);
#else
  asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
#endif

  return (eax & 6) == 6;
}

bool feature_detect_vnni() {
  uint32_t eax, ebx, ecx, edx;

// check cpu support
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  eax = cpuInfo[0];
  ebx = cpuInfo[1];
  ecx = cpuInfo[2];
  edx = cpuInfo[3];
#else
  asm volatile("cpuid\n"
               : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
               : "a"(7), "c"(0)
               : "cc");
#endif
  // avx512f  ---> 16 ebx
  // avx512dq ---> 17 ebx
  // avx512bw ---> 30 ebx
  // avx512vl ---> 31 ebx
  // avx512vnni --->11 ecx
  if (!(bit(ebx, 16) && bit(ebx, 17) && bit(ebx, 30) && bit(ebx, 31) &&
        bit(ecx, 11)))
    return false;

// check os support ymm and xmm
#if defined(_WIN32)
  eax = _xgetbv(0);
#else
  asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
#endif

  return (eax & 6) == 6;
}

bool feature_detect_avx_fma(int ftr) {
  // see Detecting Availability and Support in
  // https://software.intel.com/en-us/articles/introduction-to-intel-advanced-vector-extensions

  // check CPU support
  if (!(bit(cpuid.ecx, 27) && bit(cpuid.ecx, ftr))) return false;

  // check OS support
  uint32_t edx, eax;
#if defined(_WIN32)
  eax = _xgetbv(0);
#else
  asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
#endif

  return (eax & 6) == 6;
}

SSEType device_sse_level() {
  if (bit(cpuid.ecx, 20))
    return SSEType::ISA_SSE4_2;
  else if (bit(cpuid.ecx, 19))
    return SSEType::ISA_SSE4_1;
  else if (bit(cpuid.ecx, 0))
    return SSEType::ISA_SSE3;
  else if (bit(cpuid.edx, 26))
    return SSEType::ISA_SSE2;
  else if (bit(cpuid.edx, 25))
    return SSEType::ISA_SSE;
  else
    return SSEType::SSE_NONE;
}

AVXType device_avx_level() {
#ifdef LITE_WITH_AVX
  if (feature_detect_vnni())
    return AVXType::ISA_VNNI;
  else if (feature_detect_avx2())
    return AVXType::ISA_AVX2;
  else if (feature_detect_avx_fma(28))
    return AVXType::ISA_AVX;
  else
#endif
    return AVXType::AVX_NONE;
}

FMAType device_fma_level() {
#ifdef LITE_WITH_AVX
  if (feature_detect_avx_fma(12))
    return FMAType::ISA_FMA;
  else
#endif
    return FMAType::FMA_NONE;
}

#endif

#if defined(LITE_WITH_ANDROID) && defined(__aarch64__)
#undef AARCH64_HWCAP_SVE
#undef AARCH64_HWCAP2_SVE2
#undef AARCH64_HWCAP2_SVEAES
#undef AARCH64_HWCAP2_SVEPMULL
#undef AARCH64_HWCAP2_SVEBITPERM
#undef AARCH64_HWCAP2_SVESHA3
#undef AARCH64_HWCAP2_SVESM4
#undef AARCH64_HWCAP2_SVEI8MM
#undef AARCH64_HWCAP2_SVEF32MM
#undef AARCH64_HWCAP2_SVEF64MM
#undef AARCH64_HWCAP2_SVEBF16
#undef AARCH64_HWCAP2_I8MM
#undef AARCH64_HWCAP2_BF16
#undef AT_HWCAP
#undef AT_HWCAP2
#endif

}  // namespace lite
}  // namespace paddle
