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

#include <string>
#include <vector>

#include "lite/api/paddle_api.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/macros.h"

#ifdef LITE_WITH_METAL
#include "lite/backends/metal/target_wrapper.h"
#endif

namespace paddle {
namespace lite {

using L3CacheSetMethod = lite_api::L3CacheSetMethod;
#if defined LITE_WITH_ARM

typedef enum {
  kAPPLE = 0,
  kX1 = 1,
  kX2 = 2,
  kA35 = 35,
  kA53 = 53,
  kA55 = 55,
  kA57 = 57,
  kA510 = 60,
  kA72 = 72,
  kA73 = 73,
  kA75 = 75,
  kA76 = 76,
  kA77 = 77,
  kA78 = 78,
  kGold = 79,
  kGold_Prime = 80,
  kSilver = 81,
  kA710 = 82,
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
  bool set_a53_valid();
  bool has_sve2();
  bool has_sve2_f32mm();
  bool has_sve2_i8mm();

  void SetRunMode(lite_api::PowerMode mode, int thread_num);
  void SetCache(int l1size, int l2size, int l3size);
  void SetArch(ARMArch arch) { arch_ = arch; }

  lite_api::PowerMode mode() const { return mode_; }
  int threads() const { return active_ids_.size(); }
  ARMArch arch() const { return arch_; }
  int l1_cache_size() const { return L1_cache_[active_ids_[0]]; }
  int l2_cache_size() const { return L2_cache_[active_ids_[0]]; }
  int l3_cache_size() const { return L3_cache_[active_ids_[0]]; }

  // Methods for allocating L3Cache on Arm platform
  // Enum class L3CacheSetMethod is declared in `lite/api/paddle_api.h`
  void SetArmL3CacheSize(
      L3CacheSetMethod method = L3CacheSetMethod::kDeviceL3Cache,
      int absolute_val = -1) {
    l3_cache_method_ = method;
    absolute_l3cache_size_ = absolute_val;
    // Realloc memory for sgemm in this context.
    workspace_.clear();
    workspace_.Resize({llc_size()});
    workspace_.mutable_data<int8_t>();
  }

  void ClearArmL3Cache() { workspace_.clear(); }

  int llc_size() const {
    auto size = absolute_l3cache_size_;
    switch (l3_cache_method_) {
      // kDeviceL3Cache = 0, use the system L3 Cache size, best performance.
      case L3CacheSetMethod::kDeviceL3Cache:
        size = L3_cache_[active_ids_[0]] > 0 ? L3_cache_[active_ids_[0]]
                                             : L2_cache_[active_ids_[0]];
        break;
      // kDeviceL2Cache = 1, use the system L2 Cache size, trade off performance
      // with less memory consumption.
      case L3CacheSetMethod::kDeviceL2Cache:
        size = L2_cache_[active_ids_[0]];
        break;
      // kAbsolute = 2, use the external setting.
      case L3CacheSetMethod::kAbsolute:
        break;
      default:
        LOG(FATAL) << "Error: unknown l3_cache_method_ !";
    }
    return size > 0 ? size : 512 * 1024;
  }

  inline bool has_dot() const {
#ifdef WITH_ARM_DOTPROD
    std::vector<ARMArch> int8_arch = {kX1,
                                      kX2,
                                      kA55,
                                      kA76,
                                      kA77,
                                      kA78,
                                      kGold,
                                      kGold_Prime,
                                      kSilver,
                                      kA710,
                                      kA510};
    for (int i = 0; i < core_num_; ++i) {
      auto iter = std::find(int8_arch.begin(), int8_arch.end(), archs_[i]);
      if (iter == std::end(int8_arch)) {
        return false;
      }
    }
    return true;
#else
    return false;
#endif
  }
  bool has_fp16() const {
    std::vector<ARMArch> fp16_arch = {kX1,
                                      kX2,
                                      kA55,
                                      kA75,
                                      kA76,
                                      kA77,
                                      kA78,
                                      kGold,
                                      kGold_Prime,
                                      kSilver,
                                      kA710};
    for (int i = 0; i < core_num_; ++i) {
      auto iter = std::find(fp16_arch.begin(), fp16_arch.end(), archs_[i]);
      if (iter != std::end(fp16_arch)) {
        return true;
      }
    }
    return false;
  }

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
  bool has_a53_valid_;
  bool has_sve2_;
  bool has_sve2_i8mm_;
  bool has_sve2_f32mm_;

  // LITE_POWER_HIGH stands for using big cores,
  // LITE_POWER_LOW stands for using small core,
  // LITE_POWER_FULL stands for using all cores
  static LITE_THREAD_LOCAL lite_api::PowerMode mode_;
  static LITE_THREAD_LOCAL ARMArch arch_;
  static LITE_THREAD_LOCAL int mem_size_;
  static LITE_THREAD_LOCAL std::vector<int> active_ids_;
  static LITE_THREAD_LOCAL TensorLite workspace_;
  static LITE_THREAD_LOCAL int64_t count_;

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

  // Methods for allocating L3Cache on Arm platform
  // Enum class L3CacheSetMethod is declared in `lite/api/paddle_api.h`
  L3CacheSetMethod l3_cache_method_{L3CacheSetMethod::kDeviceL3Cache};
  int absolute_l3cache_size_{-1};
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
  static void Init(int max_stream = 6) {
    Devs& devs = Global();
    if (devs.size() > 0) {
      return;
    }
    int count = 0;
    // Get device count
    count = API::num_devices();
    if (count == 0) {
      LOG(INFO) << "No " << TargetToStr(Type) << " device(s) found!";
    } else {
      LOG(INFO) << "Found " << count << " device(s)";
    }
    CHECK_GT(max_stream, 0) << "max_stream must be greater than 0.";
    // create all device
    for (int i = 0; i < count; i++) {
      auto dev = Device<Type>(i, max_stream);
      dev.Init();
      devs.push_back(dev);
    }
    LOG(INFO) << "dev size = " << devs.size();
  }
};

#ifdef LITE_WITH_X86
enum class SSEType {
  SSE_NONE,
  ISA_SSE,
  ISA_SSE2,
  ISA_SSE3,
  ISA_SSE4_1,
  ISA_SSE4_2
};
enum class AVXType { AVX_NONE, ISA_AVX, ISA_AVX2, ISA_VNNI };
enum class FMAType { FMA_NONE, ISA_FMA };
SSEType device_sse_level();
AVXType device_avx_level();
FMAType device_fma_level();
#endif

}  // namespace lite
}  // namespace paddle
