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

#pragma once

#include <stddef.h>

#include "lite/backends/loongarch/xxl.h"

#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))

namespace paddle {
namespace lite {
namespace loongarch {

size_t CpuTotalPhysicalMemory();

//! Get the maximum allocation size for a machine.
size_t CpuMaxAllocSize();

//! Get the minimum chunk size for buddy allocator.
size_t CpuMinChunkSize();

//! Get the maximum chunk size for buddy allocator.
size_t CpuMaxChunkSize();

typedef enum {
  isa_any,
  lsx,
  lasx,
} cpu_isa_t;  // Instruction set architecture

// May I use some instruction
bool MayIUse(const cpu_isa_t cpu_isa);

}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
