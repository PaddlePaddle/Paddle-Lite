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
#include <vector>
#include "common/types.h"
#include "common/util.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype = CPU, Precision P = Precision::FP32>
class Executor {
 public:
  typedef typename PrecisionTrait<P>::ptype Ptype;
  // exector constructor
  // @param program program converted from proto program in PaddlePaddle
  // @param use_optimize bool whether use operator fusion to speed up or not
  // @param loddable bool
  Executor(const framework::Program<Dtype> program, int batch_size = 1,
           const bool use_optimize = true, const bool loddable = false);

  // predict with tensor input
  // @param t input tensor to do prediction
  // @return predicted tensor
  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t);

  // predict with lod tensor input
  // @param t input lod tensor to do prediction
  // @return predicted lod tensor
  std::shared_ptr<framework::LoDTensor> PredictLod(
      const framework::LoDTensor &t);

  // predict with vector input and dims
  // @param input vector whose elements will be formed
  // @param       input lod tensor to do prediction
  // @param dims  vector whose elements will be formed
  // @param       input tensor shape
  // @return vector which is flatted from predicted tensor
  std::vector<Ptype> Predict(const std::vector<Ptype> &input,
                             const std::vector<int64_t> &dims);

#ifdef PADDLE_MOBILE_FPGA
  void InjectVariable(const framework::Tensor &t, std::string var_name);
  void FeedData(const framework::Tensor &t);
  std::shared_ptr<framework::Tensor> FetchResult(int id = -1);
  void Predict_From_To(int start = 0, int end = -1);
  void Predict_From(int start);
  void Predict_To(int end);
#endif

 protected:
  Executor() = default;
  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t,
                                             int block_id);
  bool varInputMemory(const std::shared_ptr<framework::VarDesc> &var_desc,
                      framework::Variable *var,
                      framework::LoDTensor *tensor) const;
  void InitMemory();
  void InitCombineMemory();
  void LoadMemory(void **data,
                  const std::shared_ptr<framework::VarDesc> var_desc,
                  framework::LoDTensor *tensor);
#ifdef PADDLE_MOBILE_CL
  void LoadMemory(const framework::VarDesc var_desc, float *tensorInput,
                  char **data);
#endif
  framework::Program<Dtype> program_;
  int batch_size_ = 1;
  std::shared_ptr<framework::ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<framework::OperatorBase<Dtype>>>>
      ops_of_block_;
#ifdef PADDLE_MOBILE_PROFILE
  struct ProfInfo {
    int tid = 0;
    uint64_t runBegin = 0UL;
    uint64_t runEnd = 0UL;
  };
#endif
  bool use_optimize_ = false;
  bool loddable_ = false;
};

}  // namespace framework
}  // namespace paddle_mobile
