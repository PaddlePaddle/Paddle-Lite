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

#include <string>
#include <vector>
#include <memory.h>


#include "common/types.h"
#include "framework/tensor.h"
#include "framework/operator.h"
#include "framework/lod_tensor.h"
#include "framework/program/program.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {

template <typename Dtype, Precision P = Precision::FP32>
class Loader : PaddleMobileObject {
 public:
  const framework::Program<Dtype, P> Load(const std::string &dirname);

 private:
  void LoadVar(framework::LoDTensor *tensor, const std::string &file_path);
};

template <typename Dtype, Precision P = Precision::FP32>
class Executor {
 public:
  typedef typename PrecisionTrait<P>::ptype Ptype;

  Executor() = default;

  Executor(const framework::Program<Dtype> p);

  std::shared_ptr<framework::Tensor> predict(framework::Tensor &t);

  std::vector<Ptype> predict(const std::vector<Ptype> &input, const std::vector<int64_t> &dims);

 protected:
  void InitMemory();
  void LoadMemory(framework::LoDTensor *tensor, const std::string &file_path);
  const framework::Program<Dtype> program_;
  std::shared_ptr<framework::ProgramDesc> to_predict_program_;
  void predict(const framework::Tensor &t, int block_id);
  std::map<framework::BlockDesc,
          std::vector<std::shared_ptr<framework::OperatorBase<Dtype> >>>
  ops_of_block_;
  bool use_optimize_ = false;
};

}  // namespace paddle_mobile
