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

#include "lite/operators/sum_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SumOpLite::CheckShape() const {
  CHECK_OR_FALSE(!param_.X.empty());
  CHECK_OR_FALSE(param_.X[0]);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool SumOpLite::InferShapeImpl() const {
  if (!param_.inplace) {
    param_.Out->Resize(param_.X[0]->dims());
    param_.Out->set_lod(param_.X[0]->lod());
  }
  return true;
}

bool SumOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_names = opdesc.Input("X");
  param_.X.clear();
  for (auto input_name : X_names) {
    auto input_var = scope->FindVar(input_name);
    CHECK(input_var);
    param_.X.push_back(input_var->GetMutable<lite::Tensor>());
  }
  auto out_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(out_var);
  param_.Out = out_var->GetMutable<lite::Tensor>();
  if (opdesc.Output("Out").front() == X_names.front()) {
    param_.inplace = 1;
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sum, paddle::lite::operators::SumOpLite);
