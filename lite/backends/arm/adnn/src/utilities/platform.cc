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

bool support_sve2_i8mm() {
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

int get_cpu_num() {
#if ADNN_OS_LINUX
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
#elif ADNN_OS_IOS || ADNN_OS_MAC
  int cpu_num = 0;
  size_t len = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &len, NULL, 0);
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#else
  ADNN_LOG("get_cpu_num() is not implemented, set to default 1.\n");
  return 1;
#endif
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
#endif
  std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
  fclose(fp);
#endif
  return cpu_name;
}

size_t get_mem_size() {
#if ADNN_OS_LINUX
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
#elif ADNN_OS_IOS || ADNN_OS_MAC
  ADNN_LOG(WARNING)
      << "get_mem_size() is not implemented, set to default 4GB.\n";
  return 4096 * 1024;
#endif
  ADNN_LOG(WARNING) << "get_mem_size() is not implemented, set to default 0.\n";
  return 0;
}

void get_cpu_arch(std::vector<ARMArch>* archs, const int cpu_num) {
  archs->resize(cpu_num);
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kARMArch_UNKOWN;
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
          ADNN_LOG(ERROR) << "Unknow cpu arch: " << arch_id;
      }
      archs->at(cpu_idx) = arch_type;
      cpu_idx++;
    }
  }
  fclose(fp);
  for (; cpu_idx > 0 && cpu_idx < cpu_num; ++cpu_idx) {
    archs->at(cpu_idx) = archs->at(cpu_idx - 1);
  }
#elif ADNN_OS_IOS
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kAPPLE;
  }
#elif ADNN_OS_MAC && ADNN_ARCH_ARM64
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kX1;
  }
#endif
}

}  // namespace adnn
