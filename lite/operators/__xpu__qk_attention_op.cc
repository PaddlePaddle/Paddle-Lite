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

#include "lite/operators/__xpu__qk_attention_op.h"
#include <cmath>  // std::sqrt
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUQkAttentionOp::CheckShape() const {
  CHECK(param_.q != nullptr);
  CHECK(param_.k != nullptr);

  CHECK_EQ(param_.q->dims().size(), 3);
  CHECK_EQ(param_.k->dims().size(), 3);
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(param_.q->dims()[i], param_.k->dims()[i]);
  }
  CHECK_EQ(param_.q->dims()[2], param_.head_num * param_.head_dim);
  if (param_.mask != nullptr) {
    CHECK_EQ(param_.mask->dims().size(), 4);
    CHECK_EQ(param_.mask->dims()[0], param_.q->dims()[0]);
    CHECK_EQ(param_.mask->dims()[1], param_.head_num);
    CHECK_EQ(param_.mask->dims()[2], param_.q->dims()[1]);
    CHECK_EQ(param_.mask->dims()[3], param_.q->dims()[1]);
  }
  CHECK_EQ(param_.alpha,
           1 / std::sqrt(
                   param_.head_dim));  // For now, only support 1/sqrt(head_dim)
  return true;
}

bool XPUQkAttentionOp::InferShapeImpl() const {
  param_.output->Resize({param_.q->dims()[0],
                         param_.head_num,
                         param_.q->dims()[1],
                         param_.q->dims()[1]});
  return true;
}

bool XPUQkAttentionOp::AttachImpl(const cpp::OpDesc& op_desc,
                                  lite::Scope* scope) {
  param_.q = &scope->FindVar(op_desc.Input("q").front())->Get<lite::Tensor>();
  param_.k = &scope->FindVar(op_desc.Input("k").front())->Get<lite::Tensor>();
  param_.mask =
      &scope->FindVar(op_desc.Input("mask").front())->Get<lite::Tensor>();
  param_.output = scope->FindVar(op_desc.Output("output").front())
                      ->GetMutable<lite::Tensor>();
  param_.alpha = op_desc.GetAttr<float>("alpha");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.head_dim = op_desc.GetAttr<int>("head_dim");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__qk_attention,
                 paddle::lite::operators::XPUQkAttentionOp);
