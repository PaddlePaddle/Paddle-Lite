// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/roll_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool RollOp::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

bool RollOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  return true;
}

bool RollOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<std::vector<int64_t>>("axis");
  }
  if (opdesc.HasAttr("shifts")) {
    param_.shifts = opdesc.GetAttr<std::vector<int64_t>>("shifts");
  }

  if (opdesc.HasInput("ShiftsTensor") &&
      !opdesc.Input("ShiftsTensor").empty()) {
    auto shifts_tensor_name = opdesc.Input("ShiftsTensor").front();
    param_.ShiftsTensor =
        GetMutableVar<lite::Tensor>(scope, shifts_tensor_name);
  }

  CHECK(param_.X) << "Input(X) of RollOp should not be null.";
  CHECK(param_.Out) << "Output(Out) of RollOp should not be null.";

  input_tensor_ptrs_cache_.push_back(param_.X);
  output_tensor_ptrs_cache_.push_back(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(roll, paddle::lite::operators::RollOp);
