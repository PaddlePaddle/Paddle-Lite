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

#include "cpu_usage_info.h"
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <thread>

namespace paddle {
namespace lite_api {
namespace profile {

bool CpuUsage::IsSupported() {
#ifdef __linux__
  return true;
#endif
  return false;
}

const char* CpuUsage::get_items(const char* buffer, unsigned int item) {
  // read from buffer by offset
  const char* p = buffer;

  int len = strlen(buffer);
  int count = 0;

  for (int i = 0; i < len; i++) {
    if (' ' == *p) {
      count++;
      if (count == item - 1) {
        p++;
        break;
      }
    }
    p++;
  }
  return p;
}

unsigned long CpuUsage::get_cpu_total_occupy() {
  // get total cpu use time

  // different mode cpu occupy time
  unsigned long user_time;
  unsigned long nice_time;
  unsigned long system_time;
  unsigned long idle_time;

  FILE* fd;
  char buff[1024] = {0};

  fd = fopen("/proc/stat", "r");
  if (nullptr == fd) return 0;

  if (fgets(buff, sizeof(buff), fd) == nullptr) return 0;

  char name[64] = {0};
  sscanf(buff,
         "%s %ld %ld %ld %ld",
         name,
         &user_time,
         &nice_time,
         &system_time,
         &idle_time);
  fclose(fd);

  return (user_time + nice_time + system_time + idle_time);
}

unsigned long CpuUsage::get_cpu_proc_occupy(int pid) {
  // get specific pid cpu use time
  unsigned int tmp_pid;
  unsigned long utime;   // user time
  unsigned long stime;   // kernel time
  unsigned long cutime;  // all user time
  unsigned long cstime;  // all dead time

  char file_name[64] = {0};
  FILE* fd;
  char line_buff[1024] = {0};
  sprintf(file_name, "/proc/%d/stat", pid);

  fd = fopen(file_name, "r");
  if (nullptr == fd) return 0;

  if (fgets(line_buff, sizeof(line_buff), fd) == nullptr) return 0;

  sscanf(line_buff, "%u", &tmp_pid);
  const char* q = get_items(line_buff, process_item);
  sscanf(q, "%ld %ld %ld %ld", &utime, &stime, &cutime, &cstime);
  fclose(fd);

  return (utime + stime + cutime + cstime);
}

float CpuUsage::GetCpuUsageRatio(int pid) {
  unsigned long totalcputime1, totalcputime2;
  unsigned long procputime1, procputime2;

  totalcputime1 = get_cpu_total_occupy();
  procputime1 = get_cpu_proc_occupy(pid);

  // the 200ms is a magic number, works well
  usleep(200000);  // sleep 200ms to fetch two time point cpu usage snapshots
                   // sample for later calculation

  totalcputime2 = get_cpu_total_occupy();
  procputime2 = get_cpu_proc_occupy(pid);

  float pcpu = 0.0;
  if (0 != totalcputime2 - totalcputime1)
    pcpu = (procputime2 - procputime1) /
           float(totalcputime2 - totalcputime1);  // float number

  int cpu_num = sysconf(_SC_NPROCESSORS_ONLN);
  pcpu *= cpu_num;  // should multiply cpu num in multiple cpu machine

  return pcpu;
}

float GetCpuUsageRatio(int pid) {
  static CpuUsage cpu_monter;
  return cpu_monter.GetCpuUsageRatio(pid);
}
}  // namespace paddle
}  // namespace lite_api
}  // namespace profile
