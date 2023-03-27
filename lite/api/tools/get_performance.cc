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

#include <sched.h>
#include <string.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

class Timer {
 public:
  Timer() = default;
  virtual ~Timer() = default;

  void StartTimer() { t_start_ = std::chrono::system_clock::now(); }
  // unit ms
  float StopTimer() {
    t_stop_ = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t_stop_ -
                                                                    t_start_);
    float elapse_ms = 1000.f * static_cast<float>(ts.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den;
    return elapse_ms;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> t_start_, t_stop_;
};

int GetCpuCounts() {
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (fp == nullptr) {
    std::cout << "fopen error!" << std::endl;
    return 0;
  }
  int cnt_cpu = 0;
  char data[1024];
  while (!feof(fp)) {
    char* a = fgets(data, 1024, fp);

    if (a == nullptr) {
      break;
    }
    if (memcmp(data, "processor", 9) == 0) {
      cnt_cpu++;
    }
  }

  fclose(fp);
  fp = nullptr;
  return cnt_cpu;
}

void GetCpuFreq(int cpuid, std::vector<int>& vec) {
  char path[256];
  int freq = -1;
  // get cpuinfo_max_freq
  sprintf(
      path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
  FILE* fp = fopen(path, "rb");
  if (fp == nullptr) {
    std::cout << "cpuinfo_max_freq fopen error!" << std::endl;
    vec.emplace_back(0);
  } else {
    fscanf(fp, "%d", &freq);
    fclose(fp);
    vec.push_back(freq);
  }
  // get cpuinfo_min_freq
  sprintf(
      path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_min_freq", cpuid);
  fp = fopen(path, "rb");
  if (nullptr == fp) {
    std::cout << "cpuinfo_min_freq fopen error!" << std::endl;
    vec.emplace_back(0);
  } else {
    freq = -1;
    fscanf(fp, "%d", &freq);
    fclose(fp);
    vec.push_back(freq);
  }
}

void CpuFp32NeonFlopsTest(int32_t cnt) {
#ifdef __aarch64__
  asm volatile(
      "mov w9, %w0                     \n"
      "1:                              \n"
      "fmla v31.4s,  v31.4s,  v0.s[0]  \n"
      "fmla v30.4s,  v30.4s,  v0.s[1]  \n"
      "fmla v29.4s,  v29.4s,  v0.s[2]  \n"
      "fmla v28.4s,  v28.4s,  v0.s[3]  \n"
      "fmla v27.4s,  v27.4s,  v1.s[0]  \n"
      "fmla v26.4s,  v26.4s,  v1.s[1]  \n"
      "fmla v25.4s,  v25.4s,  v1.s[2]  \n"
      "fmla v24.4s,  v24.4s,  v1.s[3]  \n"
      "fmla v23.4s,  v23.4s,  v2.s[0]  \n"
      "fmla v22.4s,  v22.4s,  v2.s[1]  \n"
      "fmla v21.4s,  v21.4s,  v2.s[2]  \n"
      "fmla v20.4s,  v20.4s,  v2.s[3]  \n"
      "fmla v19.4s,  v19.4s,  v3.s[0]  \n"
      "fmla v18.4s,  v18.4s,  v3.s[1]  \n"
      "fmla v17.4s,  v17.4s,  v3.s[2]  \n"
      "fmla v16.4s,  v16.4s,  v3.s[3]  \n"
      "fmla v15.4s,  v15.4s,  v4.s[0]  \n"
      "fmla v14.4s,  v14.4s,  v4.s[1]  \n"
      "fmla v13.4s,  v13.4s,  v4.s[2]  \n"
      "fmla v12.4s,  v12.4s,  v4.s[3]  \n"
      "subs    w9,    w9,      #0x1    \n"
      "bne 1b                          \n"
      :
      : "r"(cnt)
      : "cc",
        "memory",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "w9");
#else
  asm volatile(
      "mov r10, %0                   \n"
      "1:                            \n"
      "vmla.f32   q15, q15, d0[0]    \n"
      "vmla.f32   q14, q14, d0[1]    \n"
      "vmla.f32   q13, q13, d1[0]    \n"
      "vmla.f32   q12, q12, d1[1]    \n"
      "vmla.f32   q11, q11, d2[0]    \n"
      "vmla.f32   q10, q10, d2[1]    \n"
      "vmla.f32   q8,  q8,  d3[1]    \n"
      "vmla.f32   q9,  q9,  d3[0]    \n"
      "vmla.f32   q7,  q7,  d4[0]    \n"
      "vmla.f32   q6,  q6,  d4[1]    \n"
      "vmla.f32   q5,  q5,  d5[0]    \n"
      "vmla.f32   q4,  q4,  d5[1]    \n"
      "subs       r10, r10, #1       \n"
      "bne        1b                 \n"
      :
      : "r"(cnt)
      : "cc",
        "memory",
        "q0",
        "q1",
        "q2",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "r10");
#endif
}

// TODO[wz1qqx] : add fp16 peak flops test
#ifdef ENABLE_ARM_FP16
void CpuFp16NeonFlopsTest(int32_t cnt) {
#ifdef __aarch64__
#else
#endif
}
#endif

void TestCPUPerformance(int32_t loop_num) {
  // get freq of cpu
  std::vector<int> FreqVec;
  for (int i = 0; i < GetCpuCounts(); i++) {
    FreqVec.clear();
    GetCpuFreq(i, FreqVec);
    std::cout << "core: " << i << ", max_freq: " << FreqVec.at(0)
              << "KHz, min_freq : " << FreqVec.at(1) << "KHz." << std::endl;
  }
  // prepare
  double flop = 0.f;
#ifdef __aarch64__
  flop = (double)loop_num * 20 * 4 * 2;
#else
  flop = (double)loop_num * 12 * 4 * 2;
#endif
// warm up
#ifdef ENABLE_ARM_FP16
  flop *= 2;
  CpuFp16NeonFlopsTest(loop_num);
#else
  CpuFp32NeonFlopsTest(loop_num);
#endif

  // compute flops
  Timer t;
  t.StartTimer();
#ifdef ENABLE_ARM_FP16
  CpuFp16NeonFlopsTest(loop_num);
#else
  CpuFp32NeonFlopsTest(loop_num);
#endif
  auto cost_ms = t.StopTimer();
  std::cout << "cost time(ms): " << cost_ms << std::endl;
  double cost_s = (double)cost_ms / 1000.0f;
  double gflop = flop / 1000000000.0f;
  float gflops = gflop / cost_s;
  std::cout << "CPU FP32 peak gflops : " << gflops << std::endl;
#ifdef __aarch64__
  std::cout << "instruction throught: tims(s) * max_freq(Hz) / (20 * "
            << loop_num << " )" << std::endl;
#else
  std::cout << "instruction throught: tims(s) * max_freq(Hz) / (12 * "
            << loop_num << " )" << std::endl;
#endif
  return;
}

int main(int argc, char** argv) {
  int cpu_id = atoi(argv[1]);
  int loop_num = atoi(argv[2]);
  std::cout << "run " << loop_num << "times loop at cpu: " << cpu_id
            << std::endl;
  // bind cpu
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_id, &cpu_set);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set) != 0) {
    std::cerr << "Error: cpu id :" << cpu_id << " bind failed!" << std::endl;
    exit(0);
  }
  // start cpu performance test
  // all cold start : using memory bandwidth
  // TODO[wz1qqx] : add more cache bandwidth test
  TestCPUPerformance(loop_num);
  return 0;
}
