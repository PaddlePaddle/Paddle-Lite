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

#include "lite/operators/sequence_arithmetic_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceArithmeticOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_EQ(param_.X->dims().size(), 2) << "Input X should a 2-D Tensor";
  CHECK_EQ(param_.Y->dims().size(), 2) << "Input Y should a 2-D Tensor";
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool SequenceArithmeticOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool SequenceArithmeticOp::AttachImpl(const cpp::OpDesc &opdesc,
                                      lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Y = scope->FindTensor(opdesc.Input("Y").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  param_.op_type = opdesc.GetAttr<int>("op_type");

  CHECK(param_.X);
  CHECK(param_.Y);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_arithmetic,
                 paddle::lite::operators::SequenceArithmeticOp);
REGISTER_LITE_OP(search_seq_arithmetic,
                 paddle::lite::operators::SequenceArithmeticOp);
