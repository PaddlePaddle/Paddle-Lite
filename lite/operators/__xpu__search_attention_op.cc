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

#include "lite/operators/__xpu__search_attention_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMmdnnSearchAttentionOp::CheckShape() const { return true; }

bool XPUMmdnnSearchAttentionOp::InferShapeImpl() const {
  auto& x_dims = param_.X->dims();
  param_.Out->Resize(x_dims);
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool XPUMmdnnSearchAttentionOp::AttachImpl(const cpp::OpDesc& op_desc,
                                           lite::Scope* scope) {
  auto x = op_desc.Input("X").front();
  auto w = op_desc.Input("W").front();
  auto b = op_desc.Input("b").front();
  auto out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.W = scope->FindVar(w)->GetMutable<lite::Tensor>();
  param_.b = scope->FindVar(b)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  param_.W_max = op_desc.GetAttr<float>("W_max");
  param_.pad_id = op_desc.GetAttr<int>("pad_id");
  param_.alpha0 = op_desc.GetAttr<float>("alpha0");
  param_.alpha1 = op_desc.GetAttr<float>("alpha1");
  param_.mask = op_desc.GetAttr<float>("mask");
  return true;
}

bool XPUMmdnnSearchAttention2Op::CheckShape() const { return true; }

bool XPUMmdnnSearchAttention2Op::InferShapeImpl() const {
  auto& x_dims = param_.X->dims();
  param_.Out->Resize(x_dims);
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool XPUMmdnnSearchAttention2Op::AttachImpl(const cpp::OpDesc& op_desc,
                                            lite::Scope* scope) {
  auto x = op_desc.Input("X").front();
  auto w = op_desc.Input("W").front();
  auto b = op_desc.Input("b").front();
  auto out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.W = scope->FindVar(w)->GetMutable<lite::Tensor>();
  param_.b = scope->FindVar(b)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  param_.W_max = op_desc.GetAttr<float>("W_max");
  param_.pad_id = op_desc.GetAttr<int>("pad_id");
  param_.alpha0 = op_desc.GetAttr<float>("alpha0");
  param_.alpha1 = op_desc.GetAttr<float>("alpha1");
  param_.mask = op_desc.GetAttr<float>("mask");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__mmdnn_search_attention,
                 paddle::lite::operators::XPUMmdnnSearchAttentionOp);

REGISTER_LITE_OP(__xpu__mmdnn_search_attention2,
                 paddle::lite::operators::XPUMmdnnSearchAttention2Op);
