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

#include "lite/operators/subgraph_op.h"
#include <utility>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SubgraphOp::CheckShape() const {
  CHECK_GE_OR_FALSE(param_.outputs.size(), 1UL);
  return true;
}

bool SubgraphOp::InferShape() const { return CheckShape(); /* enrich me */ }

bool SubgraphOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("Inputs");
  auto outputs = op_desc.Output("Outputs");
  for (auto var : inputs) {
    CHECK(scope->FindVar(var));
    param_.inputs.push_back(
        std::make_pair(var, scope->FindVar(var)->GetMutable<lite::Tensor>()));
  }
  for (auto var : outputs) {
    CHECK(scope->FindVar(var));
    param_.outputs.push_back(
        std::make_pair(var, scope->FindVar(var)->GetMutable<lite::Tensor>()));
  }
  param_.input_name_mapping =
      op_desc.GetAttr<std::vector<std::string>>("input_name_mapping");
  param_.output_name_mapping =
      op_desc.GetAttr<std::vector<std::string>>("output_name_mapping");
  CHECK(param_.sub_block);
  param_.scope = scope;
  CHECK(param_.scope);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(subgraph, paddle::lite::operators::SubgraphOp);
