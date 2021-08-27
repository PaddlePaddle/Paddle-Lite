// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/unique_with_counts_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool UniqueWithCountsOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool UniqueWithCountsOp::InferShapeImpl() const {
  DDim in_dims = param_.X->dims();
  param_.Out->Resize(in_dims);
  param_.Index->Resize(in_dims);
  param_.Count->Resize(in_dims);
  return true;
}

bool UniqueWithCountsOp::AttachImpl(const cpp::OpDesc &opdesc,
                                    lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  param_.Index = scope->FindMutableTensor(opdesc.Output("Index").front());
  param_.Count = scope->FindMutableTensor(opdesc.Output("Count").front());

  CHECK(param_.X) << "Input(X) of UniqueWithCountsOp should not be null.";
  CHECK(param_.Out) << "Output(Out) of UniqueWithCountsOp should not be null.";
  CHECK(param_.Index)
      << "Output(Index) of UniqueWithCountsOp should not be null.";
  CHECK(param_.Count)
      << "Output(Count) of UniqueWithCountsOp should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unique_with_counts,
                 paddle::lite::operators::UniqueWithCountsOp);
