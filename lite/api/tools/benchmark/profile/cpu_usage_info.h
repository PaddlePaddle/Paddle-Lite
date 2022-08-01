// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_API_TOOLS_PROFILING_CPU_USAGE_INFO_H_
#define LITE_API_TOOLS_PROFILING_CPU_USAGE_INFO_H_

#include <string>

namespace paddle {
namespace lite_api {
namespace profile {

class CpuUsage {
 public:
  float GetCpuUsageRatio(int pid);

  // Indicates whether obtaining cpu usage is supported on the platform, thus
  // indicating whether the values defined in this struct make sense or not.
  static bool IsSupported();

 private:
  const char* get_items(const char* buffer, unsigned int item);

  unsigned long get_cpu_total_occupy();

  unsigned long get_cpu_proc_occupy(int pid);

  unsigned int process_item = 14;
};

float GetCpuUsageRatio(int pid);
}  // namespace paddle
}  // namespace lite_api
}  // namespace profile

#endif  // LITE_API_TOOLS_PROFILING_CPU_USAGE_INFO_H_
