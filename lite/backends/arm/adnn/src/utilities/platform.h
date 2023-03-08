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

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(_M_IX86)
#define ADNN_ARCH_X86 1
#else
#define ADNN_ARCH_X86 0
#endif

#if defined(__x86_64__) || defined(__x86_64) || \
    defined(_M_X64) && !defined(_M_ARM64EC)
#define ADNN_ARCH_X86_64 1
#else
#define ADNN_ARCH_X86_64 0
#endif

#if defined(__arm__) || defined(_M_ARM)
#define ADNN_ARCH_ARM 1
#else
#define ADNN_ARCH_ARM 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define ADNN_ARCH_ARM64 1
#else
#define ADNN_ARCH_ARM64 0
#endif

#if defined(__ANDROID__)
#define ADNN_OS_ANDROID 1
#else
#define ADNN_OS_ANDROID 0
#endif

#if defined(__linux__)
#define ADNN_OS_LINUX 1
#else
#define ADNN_OS_LINUX 0
#endif

#if defined(__APPLE__) && TARGET_OS_IPHONE
#define ADNN_OS_IOS 1
#else
#define ADNN_OS_IOS 0
#endif

#if defined(__APPLE__) && TARGET_OS_MAC
#define ADNN_OS_MAC 1
#else
#define ADNN_OS_MAC 0
#endif

#if defined(_WIN32)
#define ADNN_OS_WINDOWS 1
#else
#define ADNN_OS_WINDOWS 0
#endif

#if defined(__QNX__)
#define ADNN_OS_QNX 1
#else
#define ADNN_OS_QNX 0
#endif

#if defined(__clang__)
#define ADNN_COMPILER_CLANG 1
#elif defined(_MSC_VER)
#define ADNN_COMPILER_MSVC 1
#elif defined(__GNUC__)
#define ADNN_COMPILER_GCC 1
#endif

#ifndef ADNN_COMPILER_CLANG
#define ADNN_COMPILER_CLANG 0
#endif

#ifndef ADNN_COMPILER_MSVC
#define ADNN_COMPILER_MSVC 0
#endif

#ifndef ADNN_COMPILER_GCC
#define ADNN_COMPILER_GCC 0
#endif

#ifndef ADNN_DISALLOW_COPY_AND_ASSIGN
#define ADNN_DISALLOW_COPY_AND_ASSIGN(class__) \
  class__(const class__&) = delete;            \
  class__& operator=(const class__&) = delete;
#endif

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
