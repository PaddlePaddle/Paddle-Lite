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
#include "common/log.h"
#include "framework/executor.h"
#include "operators/conv_op.h"
#include "operators/pool_op.h"
#include "operators/softmax_op.h"

using paddle_mobile::framework::BlockDesc;
using paddle_mobile::framework::DDim;
using paddle_mobile::framework::Executor;
using paddle_mobile::framework::LoDTensor;
using paddle_mobile::framework::OpDesc;
using paddle_mobile::framework::Program;
using paddle_mobile::framework::Tensor;
using paddle_mobile::framework::Variable;
using std::string;
template <typename DeviceType, typename OpType>
class Executor4Test : public Executor<DeviceType> {
 public:
  Executor4Test(Program<DeviceType> p, string op_type);

  std::shared_ptr<Tensor> predict(const Tensor &t, string input, string output,
                                  const DDim &dDim);
};
