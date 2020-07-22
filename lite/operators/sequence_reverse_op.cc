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

#include "lite/operators/sequence_reverse_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceReverseOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  CHECK_EQ(param_.X->lod().empty(), false)
      << "Input(X) Tensor of SequenceReverseOp does not contain "
         "LoD information.";
  CHECK_GE(param_.X->dims().size(), 2)
      << "Rank of Input(X) must be not less than 2.";
  return true;
}

bool SequenceReverseOp::InferShapeImpl() const {
  const auto *input = param_.X;
  auto out_dims = input->dims();
  param_.Out->Resize(out_dims);
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool SequenceReverseOp::AttachImpl(const cpp::OpDesc &opdesc,
                                   lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Y").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Out);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_reverse, paddle::lite::operators::SequenceReverseOp);
