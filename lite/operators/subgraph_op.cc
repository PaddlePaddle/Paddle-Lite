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

bool SubgraphOp::CheckShape() const { return true; }

bool SubgraphOp::InferShapeImpl() const { return CheckShape(); /* enrich me */ }

bool SubgraphOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input_names = op_desc.Input("Inputs");
  param_.output_names = op_desc.Output("Outputs");
  for (auto& input_name : param_.input_names) {
    CHECK(scope->FindVar(input_name));
    scope->FindVar(input_name)->GetMutable<lite::Tensor>();
  }
  for (auto& output_name : param_.output_names) {
    CHECK(scope->FindVar(output_name));
    scope->FindVar(output_name)->GetMutable<lite::Tensor>();
  }
  param_.input_data_names =
      op_desc.GetAttr<std::vector<std::string>>("input_data_names");
  param_.output_data_names =
      op_desc.GetAttr<std::vector<std::string>>("output_data_names");
  CHECK(param_.program_desc);
  param_.block_idx = op_desc.GetAttr<int32_t>("sub_block");
  CHECK_GE(param_.block_idx, 0);
  param_.exec_scope = scope;
  CHECK(param_.exec_scope);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(subgraph, paddle::lite::operators::SubgraphOp);
