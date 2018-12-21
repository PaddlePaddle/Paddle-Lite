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

#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "common/types.h"
#include "framework/executor.h"
#include "framework/load_ops.h"
#include "framework/loader.h"
#include "framework/tensor.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_engine.h"
#endif

namespace paddle_mobile {

template <typename Device, typename T = float>
class PaddleMobile {
 public:
  PaddleMobile() {
#ifndef PADDLE_MOBILE_CL
    bool is_gpu = std::is_same<DeviceType<kGPU_CL>, Device>::value;
    PADDLE_MOBILE_ENFORCE(!is_gpu, "Please recompile with GPU_CL is on");
#endif
  }
  ~PaddleMobile() {}

  PMStatus Load(const std::string &dirname, const bool optimize = false,
                const bool quantification = false, const int batch_size = 1,
                const bool lod = false);
  PMStatus Load(const std::string &model_path, const std::string &para_path,
                const bool optimize = false, const bool quantification = false,
                const int batch_size = 1, const bool lod = false);

  PMStatus Predict(const framework::Tensor &input);
  PMStatus Predict(const framework::LoDTensor &input);

  PMStatus Predict(
      const std::vector<std::pair<std::string, framework::Tensor>> &inputs);
  PMStatus Predict(
      const std::vector<std::pair<std::string, framework::LoDTensor>> &inputs);

  std::vector<T> Predict(const std::vector<T> &input,
                         const std::vector<int64_t> &dims);
  PMStatus Predict();

  void Feed(const framework::LoDTensor &input, const std::string &var_name);
  void Feed(const framework::Tensor &input, const std::string &var_name);

  typedef std::shared_ptr<framework::LoDTensor> LoDTensorPtr;
  LoDTensorPtr Fetch(const std::string &var_name);

  LoDTensorPtr Fetch() { return Fetch("fetch"); }

  bool LoadCombinedMemory(size_t model_len, const uint8_t *model_buf,
                          size_t combined_params_len,
                          uint8_t *combined_params_buf);

  void SetThreadNum(int count);
  void Clear();
  double GetPredictTime();

#ifdef PADDLE_MOBILE_FPGA
  void InjectVariable(const framework::Tensor &t, std::string var_name);
  void FeedData(const framework::Tensor &t);
  std::shared_ptr<framework::Tensor> FetchResult(int id = -1);
  void Predict_From_To(int start = 0, int end = -1);
  void Predict_From(int start);
  void Predict_To(int end);
#endif

#ifdef PADDLE_MOBILE_CL
 public:  // NOLINT
  void SetCLPath(std::string cl_path);
  int readText(const char *kernelPath,
               char **pcode);  // 读取文本文件放入 pcode，返回字符串长度
#endif

 private:
  std::shared_ptr<framework::Loader<Device, T>> loader_;
  std::shared_ptr<framework::Executor<Device, T>> executor_;
};

}  // namespace paddle_mobile
