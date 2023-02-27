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

#include "lite/operators/__xpu__qk_v_attention_op.h"
#include <cmath>  // std::sqrt
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUQkVAttentionOp::CheckShape() const {
  CHECK(param_.qk != nullptr);
  CHECK(param_.v != nullptr);

  CHECK_EQ(param_.qk->dims().size(), 4);
  CHECK_EQ(param_.v->dims().size(), 3);
  CHECK_EQ(param_.v->dims()[2], param_.head_num * param_.head_dim);
  CHECK_EQ(param_.qk->dims().size(), 4);
  CHECK_EQ(param_.qk->dims()[0], param_.v->dims()[0]);
  CHECK_EQ(param_.qk->dims()[1], param_.head_num);
  CHECK_EQ(param_.qk->dims()[2], param_.v->dims()[1]);
  CHECK_EQ(param_.qk->dims()[3], param_.v->dims()[1]);
  return true;
}

bool XPUQkVAttentionOp::InferShapeImpl() const {
  param_.output->Resize(param_.v->dims());
  return true;
}

bool XPUQkVAttentionOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  param_.qk = &scope->FindVar(op_desc.Input("qk").front())->Get<lite::Tensor>();
  param_.v = &scope->FindVar(op_desc.Input("v").front())->Get<lite::Tensor>();
  param_.output = scope->FindVar(op_desc.Output("output").front())
                      ->GetMutable<lite::Tensor>();
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.head_dim = op_desc.GetAttr<int>("head_dim");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__qk_v_attention,
                 paddle::lite::operators::XPUQkVAttentionOp);
