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
#include "framework/tensor.h"
#include "io/executor.h"
#include "io/loader.h"

namespace paddle_mobile {

template <typename Dtype = CPU, Precision P = Precision::FP32>
class PaddleMobile {
  typedef typename PrecisionTrait<P>::ptype Ptype;

 public:
  PaddleMobile() {}
  /*
   * @b load separate format fluid model
   * @b 加载分开形式的 fluid 模型
   * */
  bool Load(const std::string &dirname, bool optimize = false,
            bool quantification = false, int batch_size = 1,
            bool loddable = false);

  /*
   * @b load combine format fluid mode
   * @b 加载结合在一起格式的模型
   * */
  bool Load(const std::string &model_path, const std::string &para_path,
            bool optimize = false, bool quantification = false,
            int batch_size = 1, bool loddable = false);
  /*
   * @b 设置线程数, 当 cmake 中开启 openmp 时生效
   * */
  void SetThreadNum(int num);

  /*
   * @b to predict
   * */
  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t);

  /*
   * @b to predict
   * */
  std::shared_ptr<framework::Tensor> PredictLod(const framework::LoDTensor &t);

  /*
   * @b to predict with vector and dim
   *
   * @b 使用 输入 和 输入的维度信息 进行预测
   * */
  std::vector<Ptype> Predict(const std::vector<Ptype> &input,
                             const std::vector<int64_t> &dims);

  /**
   * 从内存加载model 以及 combinedparams的接口
   *
   * @param model_len model 文件的内存大小
   * @param model_buf model文件的内存
   * @param combined_params_len  params文件的内存大小
   * @param combined_params_buf  params文件的内存
   * @return
   */
  bool LoadCombinedMemory(size_t model_len, const uint8_t *model_buf,
                          size_t combined_params_len,
                          const uint8_t *combined_params_buf);

  void Clear();

  ~PaddleMobile();

 private:
  std::shared_ptr<Loader<Dtype, P>> loader_;
  std::shared_ptr<Executor<Dtype, P>> executor_;
};

}  // namespace paddle_mobile
