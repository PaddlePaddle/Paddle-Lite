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
#include <string>
#include <vector>

#include "framework/program/block_desc.h"
#include "framework.pb.h"
#include "operator.h"
#include "framework/program/program.h"
#include "framework/program/program_desc.h"
#include "scope.h"
#include "tensor.h"
#include "variable.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class Executor {
 public:
  Executor() = default;

  Executor(const Program<Dtype> p);

  std::shared_ptr<Tensor> predict(Tensor &t);

 public:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;

  void predict(const Tensor &t, int block_id);

  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;
};

}  // namespace framework
}  // namespace paddle_mobile
