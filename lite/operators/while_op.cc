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

#include "lite/operators/while_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool WhileOp::CheckShape() const {
  CHECK_OR_FALSE(param_.cond);
  CHECK_OR_FALSE(param_.program_desc);
  CHECK_OR_FALSE(param_.exec_scope);
  return true;
}

bool WhileOp::InferShapeImpl() const { return true; }

bool WhileOp::AttachImpl(const cpp::OpDesc &op_desc, Scope *scope) {
  auto condition = op_desc.Input("Condition");
  param_.cond = scope->FindVar(condition[0])->GetMutable<lite::Tensor>();
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

REGISTER_LITE_OP(while, paddle::lite::operators::WhileOp);
