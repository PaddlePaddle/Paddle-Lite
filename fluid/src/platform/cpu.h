
/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
#pragma once

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>

#ifdef PLATFORM_ANDROID
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

//#define gettid() syscall(SYS_gettid)

#define __NCPUBITS__ (8 * sizeof(unsigned long))

#define __CPU_SET(cpu, cpusetp)                  \
  ((cpusetp)->mask_bits[(cpu) / __NCPUBITS__] |= \
   (1UL << ((cpu) % __NCPUBITS__)))

#define __CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))
#endif

#if __APPLE__

#include "TargetConditionals.h"

#if TARGET_OS_IPHONE
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#define __IOS__
#endif
#endif

namespace padle_mobile {
namespace platform {
int getCpuCount();

int getMemInfo();

int getMaxFreq(int cpuId);

int sortBigLittleByFreq(int cpuCount, std::vector<int> &cpuIds,
                        std::vector<int> &cpuFreq,
                        std::vector<int> &clusterIds);

int setSchedAffinity(const std::vector<int> &cpuIds);

int setCpuAffinity(const std::vector<int> &cpuIds);

}  // namespace platform
}  // namespace padle_mobile
