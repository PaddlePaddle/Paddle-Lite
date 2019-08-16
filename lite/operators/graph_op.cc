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

#include "lite/operators/graph_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GraphOpLite::CheckShape() const {
  CHECK_GE_OR_FALSE(param_.inputs.size(), 1UL);
  CHECK_GE_OR_FALSE(param_.outputs.size(), 1UL);
  return true;
}

bool GraphOpLite::InferShape() const { return CheckShape(); /* enrich me */ }

bool GraphOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("Inputs");
  auto outputs = op_desc.Output("Outputs");

  for (auto var : inputs) {
    CHECK(scope->FindVar(var));
    param_.inputs.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }

  for (auto var : outputs) {
    CHECK(scope->FindVar(var));
    param_.outputs.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }

  param_.model_name = op_desc.GetAttr<std::string>("model_name");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(graph_op, paddle::lite::operators::GraphOpLite);
