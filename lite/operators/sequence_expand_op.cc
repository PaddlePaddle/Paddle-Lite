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

#include "lite/operators/sequence_expand_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceExpandOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  auto x_lod = param_.X->lod();
  auto y_lod = param_.Y->lod();
  CHECK_OR_FALSE(x_lod.size() <= 1);
  CHECK_OR_FALSE(y_lod.size() > 0);
  auto ref_level = param_.ref_level;
  CHECK_OR_FALSE(
      ref_level == -1 ||
      (ref_level >= 0 && ref_level < static_cast<int>(y_lod.size())));
  if (ref_level == -1) {
    ref_level = y_lod.size() - 1;
  }
  if (x_lod.size() > 0) {
    CHECK_EQ_OR_FALSE(x_lod[0].size(), y_lod[ref_level].size());
  }
  return true;
}

bool SequenceExpandOp::InferShapeImpl() const {
  const auto x_lod = param_.X->lod();
  auto x_dims = param_.X->dims();
  int ref_level = param_.ref_level;
  if (ref_level == -1) {
    ref_level = param_.Y->lod().size() - 1;
  }
  const auto y_lod = param_.Y->lod()[ref_level];
  auto out_dims = param_.X->dims();
  int64_t out_first_dim = 0;
  if (y_lod.size() <= 1) {
    out_first_dim = x_dims[0];
  } else {
    for (int i = 1; i < y_lod.size(); ++i) {
      int64_t x_seq_len = 1;
      if (x_lod.size() == 1) {
        x_seq_len = x_lod[0][i] - x_lod[0][i - 1];
      }
      out_first_dim += (y_lod[i] - y_lod[i - 1]) * x_seq_len;
    }
    out_dims[0] = out_first_dim;
  }
  param_.Out->Resize(out_dims);
  param_.Out->set_lod(x_lod);
  return true;
}

bool SequenceExpandOp::AttachImpl(const cpp::OpDesc &opdesc,
                                  lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Y = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("Y").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.ref_level = opdesc.GetAttr<int>("ref_level");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_expand, paddle::lite::operators::SequenceExpandOp);
