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

#include "lite/operators/where_index_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool WhereIndexdOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_GE(param_.input->dims().size(), 1);
  return true;
}

bool WhereIndexdOp::InferShapeImpl() const {
  int64_t rank = static_cast<int64_t>(param_.input->dims().size());
  int64_t numel = static_cast<int64_t>(param_.input->dims().production());
  param_.output->Resize({numel, rank});
  return true;
}

bool WhereIndexdOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  AttachParam(&param_);
  auto input = opdesc.Input("Condition").front();
  auto output = opdesc.Output("Out").front();
  CHECK(scope->FindVar(input));
  CHECK(scope->FindVar(output));
  param_.input = GetVar<lite::Tensor>(scope, input);
  param_.output = GetMutableVar<lite::Tensor>(scope, output);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(where_index, paddle::lite::operators::WhereIndexdOp);
