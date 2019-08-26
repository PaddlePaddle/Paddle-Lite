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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/types.h"
#include "common/util.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program.h"
#include "framework/tensor.h"
#include "framework/type_trait.h"
#include "pass/memory_optimize.h"

namespace paddle_mobile {
namespace framework {

template <typename Device, typename T = float>
class Executor {
 public:
  Executor(const Program<Device> &program,
           paddle_mobile::PaddleMobileConfigInternal config, int batch_size = 1,
           const bool use_optimize = true, const bool lod_mode = false);

  void SetThreadNum(int thread_num,
                    PowerMode power_mode = PERFORMANCE_PRIORITY);

  PMStatus Predict(const std::vector<std::pair<std::string, Tensor>> &inputs);
  PMStatus Predict(
      const std::vector<std::pair<std::string, LoDTensor>> &inputs);

  std::vector<T> Predict(const std::vector<T> &input,
                         const std::vector<int64_t> &dims);
  PMStatus Predict();

  void SetInput(const Tensor &input, const std::string &var_name);
  void SetInput(const LoDTensor &input, const std::string &var_name);

  std::shared_ptr<LoDTensor> GetOutput(const std::string &var_name);
#ifdef PADDLE_MOBILE_CL
  const CLImage *GetOutputImage(const std::string &var_name);
#endif

  void FeedTensorData(const std::vector<framework::Tensor> &v);
  void GetTensorResults(std::vector<framework::Tensor *> *v);
  std::string GetExceptionMsg();

#ifdef PADDLE_MOBILE_FPGA
  void InjectVariable(const Tensor &t, std::string var_name);
  void FeedData(const Tensor &t);
  void FeedData(const std::vector<void *> &v);
  void GetResults(std::vector<void *> *v);
  framework::Tensor *GetTensorByName(const std::string &name);
  std::shared_ptr<Tensor> FetchResult(int id = -1);
  void Predict_From_To(int start = 0, int end = -1);
  void Predict_From(int start);
  void Predict_To(int end);
#ifdef PADDLE_MOBILE_FPGA_V2
  void InitQuantMemory();
#endif
#endif

 protected:
  Executor() = default;

  bool varInputMemory(const std::shared_ptr<VarDesc> &var_desc,
                      Variable *var) const;
  void InitFeedFetchList();
  void InitMemory();
  void InitCombineMemory();
  void InitNoPersistableMemory(const Tensor &input_tensor);
  void LoadMemory(void **data, const std::shared_ptr<VarDesc> var_desc,
                  LoDTensor *tensor);
#ifdef PADDLE_MOBILE_CL
  void LoadMemory(const VarDesc var_desc, float *tensorInput, char **data);
#endif

  int batch_size_;
  bool use_optimize_;
  bool lod_mode_;
  PaddleMobileConfigInternal config_;
  Program<Device> program_;
  std::shared_ptr<ProgramDesc> program_desc_;
  std::vector<std::shared_ptr<OperatorBase<Device>>> ops_of_block0_;
  std::unordered_map<std::string, int> feed_indices_;
  std::unordered_map<std::string, int> fetch_indices_;
  std::string exception_msg_;

  // for super resoltion
  DDim input_dim_last_;
  bool input_dim_has_changed_ = true;

#ifdef PADDLE_MOBILE_PROFILE
  typedef typename DtypeTensorTrait<Device>::gtype ProfileTensorType;

  struct ProfInfo {
    int tid = 0;
    uint64_t runBegin = 0UL;
    uint64_t runEnd = 0UL;
  };

  void PrintProfile(const vector<Executor<Device, T>::ProfInfo> &profile) const;
#endif
};

}  // namespace framework
}  // namespace paddle_mobile
