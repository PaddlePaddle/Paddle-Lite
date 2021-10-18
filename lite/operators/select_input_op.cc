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

#include "lite/operators/select_input_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SelectInputOpLite::CheckShape() const {
  CHECK_GE_OR_FALSE(param_.X.size(), 1UL);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool SelectInputOpLite::InferShapeImpl() const {
  const std::vector<Tensor *> &inputs = param_.X;
  Tensor *&mask = param_.Mask;
  const size_t n = inputs.size();
  CHECK_GT_OR_FALSE(n, 0);

  int Mask = 0;
  Mask = *mask->data<int>();
  const auto &output_dims = inputs[Mask]->dims();
  // Set output dims
  param_.Out->Resize(output_dims);
  return true;
}

bool SelectInputOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                   lite::Scope *scope) {
  auto inputs = op_desc.Input("X");
  auto mask = op_desc.Input("Mask").front();
  auto out = op_desc.Output("Out").front();

  param_.X.clear();
  for (auto var : inputs) {
    param_.X.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  CHECK(scope->FindVar(out));
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.Mask = scope->FindVar(mask)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(select_input, paddle::lite::operators::SelectInputOpLite);
