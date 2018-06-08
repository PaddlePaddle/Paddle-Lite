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
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program.h"
#include "framework/tensor.h"

namespace paddle_mobile {

template <typename Dtype = CPU, Precision P = Precision::FP32>
class Loader {
 public:
  const framework::Program<Dtype, P> Load(const std::string &dirname,
                                          bool optimize = false);

  const framework::Program<Dtype, P> Load(const std::string &model_path,
                                          const std::string &para_path,
                                          bool optimize = false);

 private:
  const framework::Program<Dtype, P> LoadProgram(const std::string &model_path,
                                                 bool optimize = false);
  void LoadVar(framework::Variable *variable,
               const framework::VarDesc &var_desc,
               const std::string &file_path);
};

template <typename Dtype = CPU, Precision P = Precision::FP32>
class Executor {
 public:
  typedef typename PrecisionTrait<P>::ptype Ptype;

  Executor(const framework::Program<Dtype> p, int batch_size = 1,
           bool use_optimize = true);

  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t);

  std::vector<Ptype> Predict(const std::vector<Ptype> &input,
                             const std::vector<int64_t> &dims);

 protected:
  Executor() = default;

  void InitMemory();
  void LoadMemory(const framework::VarDesc var_desc,
                  framework::LoDTensor *tensor, const std::string &file_path,
                  char *data);
  void InitCombineMemory();
  framework::Program<Dtype> program_;
  int batch_size_ = 1;
  std::shared_ptr<framework::ProgramDesc> to_predict_program_;
  std::shared_ptr<framework::Tensor> Predict(const framework::Tensor &t,
                                             int block_id);
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<framework::OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;
};

}  // namespace paddle_mobile
