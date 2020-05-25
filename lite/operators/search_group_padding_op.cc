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

#include "lite/operators/search_group_padding_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchGroupPaddingOp::CheckShape() const {
  CHECK_EQ(param_.x->dims().size(), 2) << "The rank of X(Input) should be 2.";
  CHECK_EQ(param_.x->lod().empty(), false)
      << "Input Tensor of X does not contain LoD information.";
  CHECK_GE(param_.x->lod()[0].size(), 2)
      << "The Input(X)'s lod info is corrupted.";
  CHECK_EQ(param_.x->dims()[0], static_cast<int64_t>(param_.x->lod()[0].back()))
      << "The Input(X)'s lod info mismatches the actual tensor shape.";

  return true;
}

bool SearchGroupPaddingOp::InferShapeImpl() const {
  std::vector<int64_t> x_dims = param_.x->dims().Vectorize();

  param_.out_emb_padding->Resize({-1, x_dims[1]});
  param_.out_new->Resize({x_dims[0], 1});
  param_.out_padding->Resize({-1, 1});
  return true;
}

bool SearchGroupPaddingOp::AttachImpl(const cpp::OpDesc &op_desc,
                                      lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out_emb_padding = op_desc.Output("Out_emb_padding").front();
  auto out_new = op_desc.Output("Out_new").front();
  auto out_padding = op_desc.Output("Out_padding").front();

  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.out_emb_padding =
      scope->FindVar(out_emb_padding)->GetMutable<lite::Tensor>();
  param_.out_new = scope->FindVar(out_new)->GetMutable<lite::Tensor>();
  param_.out_padding = scope->FindVar(out_padding)->GetMutable<lite::Tensor>();
  param_.pad_id = op_desc.GetAttr<int>("pad_id");

  CHECK(param_.out_emb_padding)
      << "Output(Out_emb_padding) of SearchGroupPadding Op should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(search_group_padding,
                 paddle::lite::operators::SearchGroupPaddingOp);
