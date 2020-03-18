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

#include <cstdarg>
#include <string>
#include <vector>
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"
#ifdef LITE_WITH_MLU
#include "lite/backends/mlu/mlu_utils.h"
#endif

namespace paddle {
namespace lite {

#if ((defined LITE_WITH_ARM) || (defined LITE_WITH_MLU))

typedef enum {
  kAPPLE = 0,
  kA53 = 53,
  kA55 = 55,
  kA57 = 57,
  kA72 = 72,
  kA73 = 73,
  kA75 = 75,
  kA76 = 76,
  kARMArch_UNKOWN = -1
} ARMArch;

class DeviceInfo {
 public:
  static DeviceInfo& Global() {
    static auto* x = new DeviceInfo;
    return *x;
  }

  static int Init() {
    static int ret = Global().Setup();
    return ret;
  }

  int Setup();

  void SetRunMode(lite_api::PowerMode mode, int thread_num);
#ifdef LITE_WITH_MLU
  void SetMLURunMode(lite_api::MLUCoreVersion core_version,
                     int core_number,
                     bool use_first_conv,
                     const std::vector<float>& mean_vec,
                     const std::vector<float>& std_vec);
  cnmlCoreVersion_t MLUCoreVersion();
  int MLUCoreNumber();
  bool UseFirstConv();
  const std::vector<float>& MeanVec() const;
  const std::vector<float>& StdVec() const;
#endif
  void SetCache(int l1size, int l2size, int l3size);
  void SetArch(ARMArch arch) { arch_ = arch; }

  lite_api::PowerMode mode() const { return mode_; }
  int threads() const { return active_ids_.size(); }
  ARMArch arch() const { return arch_; }
  int l1_cache_size() const { return L1_cache_[active_ids_[0]]; }
  int l2_cache_size() const { return L2_cache_[active_ids_[0]]; }
  int l3_cache_size() const { return L3_cache_[active_ids_[0]]; }
  int llc_size() const {
    auto size = L3_cache_[active_ids_[0]] > 0 ? L3_cache_[active_ids_[0]]
                                              : L2_cache_[active_ids_[0]];
    return size > 0 ? size : 512 * 1024;
  }
  bool has_dot() const { return dot_[active_ids_[0]]; }
  bool has_fp16() const { return fp16_[active_ids_[0]]; }

  template <typename T>
  T* workspace_data() {
    return reinterpret_cast<T*>(workspace_.mutable_data<int8_t>());
  }
  bool ExtendWorkspace(size_t size);

 private:
  int core_num_;
  std::vector<int> max_freqs_;
  std::vector<int> min_freqs_;
  std::string dev_name_;

  std::vector<int> L1_cache_;
  std::vector<int> L2_cache_;
  std::vector<int> L3_cache_;
  std::vector<int> core_ids_;
  std::vector<int> big_core_ids_;
  std::vector<int> little_core_ids_;
  std::vector<int> cluster_ids_;
  std::vector<ARMArch> archs_;
  std::vector<bool> fp32_;
  std::vector<bool> fp16_;
  std::vector<bool> dot_;

  // LITE_POWER_HIGH stands for using big cores,
  // LITE_POWER_LOW stands for using small core,
  // LITE_POWER_FULL stands for using all cores
  static thread_local lite_api::PowerMode mode_;
  static thread_local ARMArch arch_;
  static thread_local int mem_size_;
  static thread_local std::vector<int> active_ids_;
  static thread_local TensorLite workspace_;
  static thread_local int64_t count_;

#ifdef LITE_WITH_MLU
  static thread_local cnmlCoreVersion_t mlu_core_version_;
  static thread_local int mlu_core_number_;
  static thread_local bool use_first_conv_;
  static thread_local std::vector<float> mean_vec_;
  static thread_local std::vector<float> std_vec_;
#endif

  void SetDotInfo(int argc, ...);
  void SetFP16Info(int argc, ...);
  void SetFP32Info(int argc, ...);
  void SetCacheInfo(int cache_id, int argc, ...);
  void SetArchInfo(int argc, ...);
  bool SetCPUInfoByName();
  void SetCPUInfoByProb();
  void RequestPowerFullMode(int thread_num);
  void RequestPowerHighMode(int thread_num);
  void RequestPowerLowMode(int thread_num);
  void RequestPowerNoBindMode(int thread_num);
  void RequestPowerRandHighMode(int shift_num, int thread_num);
  void RequestPowerRandLowMode(int shift_num, int thread_num);

  DeviceInfo() = default;
};
#endif  // LITE_WITH_ARM

template <TargetType Type>
class Device;

template <TargetType Type>
class Env {
 public:
  typedef TargetWrapper<Type> API;
  typedef std::vector<Device<Type>> Devs;
  static Devs& Global() {
    static Devs* devs = new Devs();
    return *devs;
  }
  static void Init(int max_stream = 4) {
#ifdef LITE_WITH_MLU
    CNRT_CALL(cnrtInit(0));
#endif
    Devs& devs = Global();
    if (devs.size() > 0) {
      return;
    }
    int count = 0;
    // Get device count
    count = API::num_devices();
    if (count == 0) {
      CHECK(false) << "No device found!";
    } else {
      LOG(INFO) << "Found " << count << " device(s)";
    }
    // create all device
    for (int i = 0; i < count; i++) {
      auto dev = Device<Type>(i, max_stream);
      dev.Init();
      devs.push_back(dev);
    }
    LOG(INFO) << "dev size = " << devs.size();
  }
};

#ifdef LITE_WITH_MLU
void SetMluDevice(int device_id);

template <>
class Device<TARGET(kMLU)> {
 public:
  Device(int dev_id, int max_queue = 1) : idx_(dev_id), max_queue_(max_queue) {}
  void Init();

  int id() { return idx_; }
  int max_queue() { return max_queue_; }
  void SetId(int idx) { idx_ = idx; }
  std::string name() { return "MLU"; }
  int core_num() { return 16; }
  float max_memory() { return 16 * 1024; }
  std::vector<cnrtQueue_t> io_queues() { return io_queue_; }
  std::vector<cnrtQueue_t> exec_queues() { return exec_queue_; }

 private:
  void CreateQueue();
  void GetInfo();

 private:
  int idx_{0};
  int max_queue_;
  std::string device_name_;
  float max_memory_;

  std::vector<cnrtQueue_t> io_queue_;
  std::vector<cnrtQueue_t> exec_queue_;
};

template class Env<TARGET(kMLU)>;
#endif  // LITE_WITH_MLU

#ifdef LITE_WITH_CUDA
template <>
class Device<TARGET(kCUDA)> {
 public:
  Device(int dev_id, int max_stream = 1)
      : idx_(dev_id), max_stream_(max_stream) {}
  void Init();

  int id() { return idx_; }
  int max_stream() { return max_stream_; }
  void SetId(int idx) { idx_ = idx; }
  std::string name() { return device_prop_.name; }
  int core_num() { return device_prop_.multiProcessorCount; }
  float max_memory() { return device_prop_.totalGlobalMem / 1048576.; }
  std::vector<cudaStream_t> exec_streams() { return exec_stream_; }
  std::vector<cudaStream_t> io_streams() { return io_stream_; }

  int sm_version() { return sm_version_; }
  bool has_fp16() { return has_fp16_; }
  bool has_int8() { return has_fp16_; }
  bool has_hmma() { return has_fp16_; }
  bool has_imma() { return has_fp16_; }
  int runtime_version() { return runtime_version_; }

 private:
  void CreateStream();
  void GetInfo();

 private:
  int idx_{0};
  int max_stream_;
  cudaDeviceProp device_prop_;
  std::string device_name_;
  float max_memory_;

  int sm_version_;
  bool has_fp16_;
  bool has_int8_;
  bool has_hmma_;
  bool has_imma_;
  int runtime_version_;
  std::vector<cudaStream_t> exec_stream_;
  std::vector<cudaStream_t> io_stream_;
};

template class Env<TARGET(kCUDA)>;
#endif

}  // namespace lite
}  // namespace paddle
