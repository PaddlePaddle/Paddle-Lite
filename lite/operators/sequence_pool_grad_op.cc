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

#include "lite/operators/sequence_pool_grad_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequencePoolGradOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.X_Grad);
  CHECK_OR_FALSE(param_.Out_Grad);
  auto lod = param_.X->lod();
  CHECK_EQ_OR_FALSE(lod.size(), 1UL);
  auto dims = param_.X->dims();
  CHECK_GE_OR_FALSE(dims[0], (static_cast<int64_t>(lod[0].size()) - 1));
  return true;
}

bool SequencePoolGradOp::InferShapeImpl() const {
  const auto *input = param_.X;
  auto x_dims = input->dims();
  if (param_.X_Grad) {
    param_.X_Grad->Resize(x_dims);
    param_.X_Grad->set_lod(param_.X->lod());
  }
  return true;
}

bool SequencePoolGradOp::AttachImpl(const cpp::OpDesc &opdesc,
                                    lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  CHECK(param_.X);
  auto *out_grad_var = scope->FindVar(opdesc.Input("Out@GRAD").front());
  CHECK(out_grad_var);
  param_.Out_Grad = &out_grad_var->Get<Tensor>();

  auto *x_grad_var = scope->FindVar(opdesc.Output("X@GRAD").front());
  CHECK(x_grad_var);
  param_.X_Grad = x_grad_var->GetMutable<Tensor>();

  param_.pool_type = opdesc.GetAttr<std::string>("pooltype");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_pool_grad,
                 paddle::lite::operators::SequencePoolGradOp);
