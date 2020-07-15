// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/sequence_mask_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceMaskOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  return true;
}

bool SequenceMaskOp::InferShapeImpl() const { return true; }

bool SequenceMaskOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  if (opdesc.HasInput("MaxLenTensor") &&
      !opdesc.Input("MaxLenTensor").empty()) {
    auto var = scope->FindVar(opdesc.Input("MaxLenTensor").front());
    if (var != nullptr) {
      param_.MaxLenTensor = var->GetMutable<lite::Tensor>();
    }
  }
  param_.Y =
      scope->FindVar(opdesc.Output("Y").front())->GetMutable<lite::Tensor>();
  param_.maxlen = opdesc.GetAttr<int>("maxlen");
  param_.out_dtype = opdesc.GetAttr<int>("out_dtype");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_mask, paddle::lite::operators::SequenceMaskOp);
