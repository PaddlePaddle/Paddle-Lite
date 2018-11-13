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

template <typename Dtype = CPU, Precision P = Precision::FP32>
class PaddleMobile {
  typedef typename PrecisionTrait<P>::ptype Ptype;

 public:
  PaddleMobile() {
#ifndef PADDLE_MOBILE_CL
    bool is_gpu = std::is_same<DeviceType<kGPU_CL>, Dtype>::value;
    PADDLE_MOBILE_ENFORCE(!is_gpu,
                          "Not Enable GPU in CmakeList but run gpu codes ");
#endif
  }
  bool Load(const std::string &dirname, bool optimize = false,
            bool quantification = false, int batch_size = 1,
            bool loddable = false);

  bool Load(const std::string &model_path, const std::string &para_path,
            bool optimize = false, bool quantification = false,
            int batch_size = 1, bool loddable = false);

  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t);

  std::shared_ptr<framework::Tensor> PredictLod(const framework::LoDTensor &t);

  std::vector<Ptype> Predict(const std::vector<Ptype> &input,
                             const std::vector<int64_t> &dims);

  bool LoadCombinedMemory(size_t model_len, const uint8_t *model_buf,
                          size_t combined_params_len,
                          uint8_t *combined_params_buf);

  void SetThreadNum(int num);
  void Clear();
  double GetPredictTime();

  ~PaddleMobile();

#ifdef PADDLE_MOBILE_FPGA
  void InjectVariable(const framework::Tensor &t, std::string var_name);
  void FeedData(const framework::Tensor &t);
  std::shared_ptr<framework::Tensor> FetchResult(int id = -1);
  void Predict_From_To(int start = 0, int end = -1);
  void Predict_From(int start);
  void Predict_To(int end);
#endif

#ifdef PADDLE_MOBILE_CL
 public:
  void SetCLPath(std::string cl_path);
<<<<<<< HEAD
=======
  int readText(const char *kernelPath,
               char **pcode);  // 读取文本文件放入 pcode，返回字符串长度
>>>>>>> upstream/develop
#endif

 private:
  std::shared_ptr<framework::Loader<Dtype, P>> loader_;
  std::shared_ptr<framework::Executor<Dtype, P>> executor_;
};

}  // namespace paddle_mobile
