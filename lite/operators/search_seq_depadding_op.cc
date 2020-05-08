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

#include "lite/operators/search_seq_depadding_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchSeqDepaddingOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.pad);
  CHECK_OR_FALSE(param_.src);
  CHECK_OR_FALSE(param_.out);

  DDim pad_dims = param_.pad->dims();
  DDim src_dims = param_.src->dims();
  CHECK_OR_FALSE(pad_dims.size() == 2);
  CHECK_OR_FALSE(src_dims.size() == 2);

  const auto& pad_lod = param_.pad->lod();
  CHECK_OR_FALSE(!pad_lod.empty());
  const auto& pad_lod_0 = pad_lod[0];
  CHECK_OR_FALSE(pad_lod_0.size() >= 2);
  CHECK_OR_FALSE(pad_dims[0] == pad_lod_0.back());

  const auto& src_lod = param_.src->lod();
  CHECK_OR_FALSE(!src_lod.empty());
  const auto& src_lod_0 = src_lod[0];
  CHECK_OR_FALSE(src_lod_0.size() >= 2);
  CHECK_OR_FALSE(src_dims[0] == src_lod_0.back());
  return true;
}

bool SearchSeqDepaddingOpLite::InferShapeImpl() const {
  DDim pad_dims = param_.pad->dims();
  param_.out->Resize({-1, pad_dims[1]});
  return true;
}

bool SearchSeqDepaddingOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                          lite::Scope* scope) {
  auto pad = op_desc.Input("Pad").front();
  auto src = op_desc.Input("Src").front();
  auto out = op_desc.Output("Out").front();

  param_.pad = scope->FindVar(pad)->GetMutable<lite::Tensor>();
  param_.src = scope->FindVar(src)->GetMutable<lite::Tensor>();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(search_seq_depadding,
                 paddle::lite::operators::SearchSeqDepaddingOpLite);
