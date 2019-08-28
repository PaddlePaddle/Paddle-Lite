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
__attribute__((__visibility__("default"))) enum DeviceTypeEnum {
  kCPU = 0,
  kGPU_CL = 1
};

// ddim class
class DDim {
 public:
  __attribute__((__visibility__("default"))) int size();
  __attribute__((__visibility__("default"))) int64_t &operator[](int idx);
  __attribute__((__visibility__("default"))) int64_t operator[](int idx) const;

  __attribute__((__visibility__("default"))) std::vector<int64_t> dims;
};
__attribute__((__visibility__("default"))) DDim make_ddim(
    const std::vector<int64_t> &dims);

// tensor class
class Tensor {
 public:
  __attribute__((__visibility__("default"))) Tensor(float *data, DDim ddim);

  __attribute__((__visibility__("default"))) float *data() const;
  __attribute__((__visibility__("default"))) DDim dims() const;

 private:
  float *data_;
  DDim ddim_;
};

// net class
class Net {
 public:
  __attribute__((__visibility__("default"))) Net(DeviceTypeEnum device);
  __attribute__((__visibility__("default"))) ~Net();
  __attribute__((__visibility__("default"))) void SetThreadNum(int thread_num);
  __attribute__((__visibility__("default"))) void SetCLPath(std::string path);
  __attribute__((__visibility__("default"))) bool Load(
      const std::string &dirname, const bool optimize = false,
      const bool quantification = false, const int batch_size = 1,
      const bool lod_mode = false);
  __attribute__((__visibility__("default"))) bool Load(
      const std::string &model_path, const std::string &para_path,
      const bool optimize = false, const bool quantification = false,
      const int batch_size = 1, const bool lod_mode = false);
  __attribute__((__visibility__("default"))) bool LoadCombinedMemory(
      size_t model_len, const uint8_t *model_buf, size_t combined_params_len,
      uint8_t *combined_params_buf, bool optimize = false,
      bool quantification = false, int batch_size = 1, bool lod_mode = false);
  __attribute__((__visibility__("default"))) std::vector<float> Predict(
      const std::vector<float> &input, const std::vector<int64_t> &dims);
  __attribute__((__visibility__("default"))) bool Predict();
  __attribute__((__visibility__("default"))) bool Predict(const Tensor &input);
  __attribute__((__visibility__("default"))) void Feed(
      const std::string &var_name, const Tensor &input);
  __attribute__((__visibility__("default"))) std::shared_ptr<Tensor> Fetch(
      const std::string &var_name);

 private:
  void *engine_ = nullptr;
  DeviceTypeEnum device_;
};

#endif

}  // namespace wrap
}  // namespace paddle_mobile
