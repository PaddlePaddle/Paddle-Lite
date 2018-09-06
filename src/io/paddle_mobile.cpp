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

#include "io/paddle_mobile.h"

namespace paddle_mobile {

template <typename Dtype, Precision P>
void PaddleMobile<Dtype, P>::SetThreadNum(int num) {
#ifdef _OPENMP
  //  omp_set_dynamic(0);
  omp_set_num_threads(num);
#endif
};

template <typename Dtype, Precision P>
bool PaddleMobile<Dtype, P>::Load(const std::string &dirname, bool optimize,
                                  bool quantification, int batch_size,
                                  bool loddable) {
  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<Loader<Dtype, P>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }

  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<Executor<Dtype, P>>(
        loader_->Load(dirname, optimize, quantification), batch_size, optimize,
        loddable);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return true;
}

template <typename Dtype, Precision P>
bool PaddleMobile<Dtype, P>::Load(const std::string &model_path,
                                  const std::string &para_path, bool optimize,
                                  bool quantification, int batch_size,
                                  bool loddable) {
  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<Loader<Dtype, P>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }

  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<Executor<Dtype, P>>(
        loader_->Load(model_path, para_path, optimize, quantification),
        batch_size, optimize, loddable);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return true;
}

template <typename Dtype, Precision P>
bool PaddleMobile<Dtype, P>::LoadCombinedMemory(
    size_t model_len, const uint8_t *model_buf, size_t combined_params_len,
    const uint8_t *combined_params_buf) {
  int batch_size = 1;
  bool optimise = true;
  bool quantification = false;

  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<Loader<Dtype, P>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }

  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<Executor<Dtype, P>>(
        loader_->LoadCombinedMemory(model_len, model_buf, combined_params_len,
                                    combined_params_buf, optimise,
                                    quantification),
        batch_size, optimise);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return true;
}
template <typename Dtype, Precision P>
std::shared_ptr<framework::Tensor> PaddleMobile<Dtype, P>::Predict(
    const framework::Tensor &t) {
  return executor_->Predict(t);
}

template <typename Dtype, Precision P>
std::shared_ptr<framework::Tensor> PaddleMobile<Dtype, P>::PredictLod(
    const framework::LoDTensor &t) {
  return executor_->PredictLod(t);
}

template <typename Dtype, Precision P>
std::vector<typename PaddleMobile<Dtype, P>::Ptype>
PaddleMobile<Dtype, P>::Predict(const std::vector<Ptype> &input,
                                const std::vector<int64_t> &dims) {
  return executor_->Predict(input, dims);
}

template <typename Dtype, Precision P>
void PaddleMobile<Dtype, P>::Clear() {
  executor_ = nullptr;
  loader_ = nullptr;
}

template <typename Dtype, Precision P>
PaddleMobile<Dtype, P>::~PaddleMobile() {
  executor_ = nullptr;
  loader_ = nullptr;
}

template class PaddleMobile<CPU, Precision::FP32>;

template class PaddleMobile<FPGA, Precision::FP32>;

template class PaddleMobile<GPU_MALI, Precision::FP32>;

}  // namespace paddle_mobile
