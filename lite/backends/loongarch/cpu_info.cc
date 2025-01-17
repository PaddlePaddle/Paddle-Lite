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

#include "lite/backends/loongarch/cpu_info.h"

#include <unistd.h>

#include <sys/auxv.h> // For getauxval
#ifndef HWCAP_LOONGARCH_LSX
#define HWCAP_LOONGARCH_LSX             (1 << 4)
#endif
#ifndef HWCAP_LOONGARCH_LASX
#define HWCAP_LOONGARCH_LASX            (1 << 5)
#endif

#include <algorithm>
#include "lite/utils/log/cp_logging.h"

#include "lite/utils/env.h"

// DEFINE_double(fraction_of_cpu_memory_to_use,
//               1,
//               "Default use 100% of CPU memory for PaddlePaddle,"
//               "reserve the rest for page tables, etc");
double fraction_of_cpu_memory_to_use =
    paddle::lite::GetDoubleFromEnv("fraction_of_cpu_memory_to_use", 1);

// DEFINE_uint64(initial_cpu_memory_in_mb,
//               500ul,
//               "Initial CPU memory for PaddlePaddle, in MD unit.");
uint64_t initial_cpu_memory_in_mb =
    paddle::lite::GetUInt64FromEnv("initial_cpu_memory_in_mb", 500ul);

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
// DEFINE_bool(use_pinned_memory, true, "If set, allocate cpu pinned memory.");
bool use_pinned_memory =
    paddle::lite::GetBoolFromEnv("use_pinned_memory", true);

namespace paddle {
namespace lite {
namespace loongarch {

size_t CpuTotalPhysicalMemory() {
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

size_t CpuMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return fraction_of_cpu_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

size_t CpuMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 3% of CPU memory,
  // or the initial_cpu_memory_in_mb.
  return std::min(static_cast<size_t>(CpuMaxAllocSize() / 32),
                  static_cast<size_t>(initial_cpu_memory_in_mb * 1 << 20));
}

bool MayIUse(const cpu_isa_t cpu_isa) {
  static long loong_hwcap = 0UL; // HWCAP should not be zero usually.
  if (loong_hwcap == 0UL)
    loong_hwcap = getauxval(AT_HWCAP);
  switch (cpu_isa) {
    case lsx:
      return (loong_hwcap & HWCAP_LOONGARCH_LSX) != 0;
    case lasx:
      return (loong_hwcap & HWCAP_LOONGARCH_LASX) != 0;
    case isa_any:
      return true;
    default:
      return false;
  }
}

}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
