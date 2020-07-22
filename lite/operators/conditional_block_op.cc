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

#include "lite/operators/conditional_block_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConditionalBlockOp::CheckShape() const {
  CHECK_OR_FALSE(param_.cond);
  CHECK_OR_FALSE(param_.program_desc);
  CHECK_OR_FALSE(param_.exec_scope);
  return true;
}

bool ConditionalBlockOp::InferShapeImpl() const { return true; }

bool ConditionalBlockOp::AttachImpl(const cpp::OpDesc& op_desc, Scope* scope) {
  auto condition = op_desc.Input("Cond").front();
  param_.cond = scope->FindVar(condition)->GetMutable<lite::Tensor>();
  auto inputs = op_desc.Input("Input");
  for (const auto& input : inputs) {
    auto* var = scope->FindVar(input);
    CHECK(var);
    param_.inputs.push_back(var->GetMutable<lite::Tensor>());
  }
  auto outs = op_desc.Output("Out");
  for (const auto& out : outs) {
    auto* var = scope->FindVar(out);
    CHECK(var);
    param_.outs.push_back(var->GetMutable<lite::Tensor>());
  }
  param_.is_scalar_condition = op_desc.GetAttr<bool>("is_scalar_condition");
  // obtain sub_block in core program.cc
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

REGISTER_LITE_OP(conditional_block,
                 paddle::lite::operators::ConditionalBlockOp);
