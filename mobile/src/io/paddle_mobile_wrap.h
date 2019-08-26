/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace paddle_mobile {
namespace wrap {

#ifndef PADDLE_MOBILE_FPGA

// device type
enum DeviceTypeEnum {
  kINVALID = -1,
  kCPU = 0,
  kFPGA = 1,
  kGPU_MALI = 2,
  kGPU_CL = 3
};

template <DeviceTypeEnum T>
struct DeviceType {};

typedef DeviceType<kCPU> CPU;
typedef DeviceType<kFPGA> FPGA;
typedef DeviceType<kGPU_MALI> GPU_MALI;
typedef DeviceType<kGPU_CL> GPU_CL;

// ddim class
class DDim {
 public:
  int size();
  int64_t &operator[](int idx);
  int64_t operator[](int idx) const;

  std::vector<int64_t> dims;
};
DDim make_ddim(const std::vector<int64_t> &dims);

// tensor class
class Tensor {
 public:
  Tensor(float *data, DDim ddim);

  template <typename T>
  float *data() const;
  DDim dims() const;

  float *data_;
  DDim ddim_;
};

// pm status
enum PMStatus {
  PMSuccess = 0xFF,        /*!< No errors */
  PMNotInitialized = 0x01, /*!< Data not initialized. */
  PMInvalidValue = 0x02,   /*!< Incorrect variable value. */
  PMMemAllocFailed = 0x03, /*!< Memory allocation error. */
  PMUnKownError = 0x04,    /*!< Unknown error. */
  PMOutOfAuthority = 0x05, /*!< Try to modified data not your own*/
  PMOutOfMem = 0x06,       /*!< OOM error*/
  PMUnImplError = 0x07,    /*!< Unimplement error. */
  PMWrongDevice = 0x08     /*!< un-correct device. */
};

// net class
template <typename Device>
class Net {
 public:
  Net();
  ~Net();
  void SetThreadNum(int thread_num);
  PMStatus Load(const std::string &dirname, const bool optimize = false,
                const bool quantification = false, const int batch_size = 1,
                const bool lod_mode = false);
  PMStatus Load(const std::string &model_path, const std::string &para_path,
                const bool optimize = false, const bool quantification = false,
                const int batch_size = 1, const bool lod_mode = false);
  bool LoadCombinedMemory(size_t model_len, const uint8_t *model_buf,
                          size_t combined_params_len,
                          uint8_t *combined_params_buf, bool optimize = false,
                          bool quantification = false, int batch_size = 1,
                          bool lod_mode = false);
  PMStatus Predict(const Tensor &input);
  std::vector<float> Predict(const std::vector<float> &input,
                             const std::vector<int64_t> &dims);
  PMStatus Predict();
  void Feed(const std::string &var_name, const Tensor &input);
  std::shared_ptr<Tensor> Fetch(const std::string &var_name);
  void *engine_ = nullptr;
};

#endif

}  // namespace wrap
}  // namespace paddle_mobile
