// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/operators/stack_op.h"
#include <cstddef>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool StackOp::CheckShape() const {
  auto input = param_.X;
  for (auto x : input) {
    CHECK_OR_FALSE(x);
  }
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool StackOp::InferShapeImpl() const {
  auto input = param_.X;
  auto input_dims = input[0]->dims();
  int axis = param_.axis;
  int rank = input_dims.size();
  if (axis < 0) axis += (rank + 1);
  auto vec = input_dims.Vectorize();
  vec.insert(vec.begin() + axis, input.size());
  param_.Out->Resize(vec);
  return true;
}

bool StackOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto X = op_desc.Input("X");
  auto Out = op_desc.Output("Y").front();
  param_.X.clear();
  for (auto var : X) {
    param_.X.emplace_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  param_.axis = op_desc.GetAttr<int>("axis");
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(stack, paddle::lite::operators::StackOp);
