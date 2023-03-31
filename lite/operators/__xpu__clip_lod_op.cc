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

#include "lite/operators/__xpu__clip_lod_op.h"
#include <cmath>  // std::sqrt
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUClipLodOp::CheckShape() const {
  CHECK(param_.SeqLod != nullptr);
  CHECK(param_.SeqLen != nullptr);
  CHECK(param_.PadSeqLen != nullptr);

  CHECK_EQ(param_.SeqLod->dims().size(), 1);
  CHECK_EQ(param_.SeqLen->dims().size(), 1);
  CHECK_EQ(param_.PadSeqLen->dims().size(), 1);

  CHECK_EQ(param_.SeqLod->dims()[0], param_.SeqLen->dims()[0] + 1);

  return true;
}

bool XPUClipLodOp::InferShapeImpl() const {
  param_.NewSeqLod->Resize(param_.SeqLod->dims());
  param_.NewSeqLen->Resize(param_.SeqLen->dims());
  param_.NewPadSeqLen->Resize({1});
  return true;
}

bool XPUClipLodOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.SeqLod =
      &scope->FindVar(op_desc.Input("SeqLod").front())->Get<lite::Tensor>();
  param_.SeqLen =
      &scope->FindVar(op_desc.Input("SeqLen").front())->Get<lite::Tensor>();
  param_.PadSeqLen =
      &scope->FindVar(op_desc.Input("PadSeqLen").front())->Get<lite::Tensor>();
  param_.NewSeqLod =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("NewSeqLod").front());
  param_.NewSeqLen =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("NewSeqLen").front());
  param_.NewPadSeqLen = GetMutableVar<lite::Tensor>(
      scope, op_desc.Output("NewPadSeqLen").front());
  param_.keep_ratio = op_desc.GetAttr<float>("keep_ratio");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__clip_lod, paddle::lite::operators::XPUClipLodOp);
