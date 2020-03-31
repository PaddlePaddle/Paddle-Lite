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

#include "lite/operators/search_seq_fc_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchSeqFcOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.w);
  CHECK_OR_FALSE(param_.out);
  return true;
}

bool SearchSeqFcOpLite::InferShapeImpl() const {
  const auto x_dims = param_.x->dims();
  const auto w_dims = param_.w->dims();
  const auto& x_lod = param_.x->lod();
  auto out_size = param_.out_size;
  CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
  CHECK(!x_lod.empty()) << "The Input(X) must hold lod info.";
  const auto& x_lod_0 = x_lod[0];
  CHECK_GE(x_lod_0.size(), 2) << "The Input(X)'s lod info is corrupted.";
  CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod_0.back()))
      << "The Input(X)'s lod info mismatches the actual tensor shape.";
  CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
  CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
  CHECK_EQ(w_dims[0], out_size) << "Wrong shape: w_dims[0] != out_size";

  if (param_.b != nullptr) {
    const auto b_dims = param_.b->dims();
    CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
    CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
  }

  param_.out->set_lod(x_lod);
  param_.out->Resize({x_dims[0], w_dims[0]});
  return true;
}

bool SearchSeqFcOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("W").empty());
  CHECK(!op_desc.Output("Out").empty());
  auto x = op_desc.Input("X").front();
  auto w = op_desc.Input("W").front();
  auto out = op_desc.Output("Out").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.w = scope->FindVar(w)->GetMutable<lite::Tensor>();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.out_size = op_desc.GetAttr<int>("out_size");
  bool has_bias = op_desc.GetAttr<bool>("has_bias");
  if (has_bias) {
    CHECK(!op_desc.Input("b").empty());
    auto b = op_desc.Input("b").front();
    param_.b = scope->FindVar(b)->GetMutable<lite::Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(search_seq_fc, paddle::lite::operators::SearchSeqFcOpLite);
