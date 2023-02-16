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

#include "lite/operators/calculate_lod_op.h"

namespace paddle {
namespace lite {
namespace operators {

bool CalculateLodOp::CheckShape() const {
  CHECK(param_.Mask != nullptr);
  CHECK(param_.SeqLod != nullptr);
  CHECK(param_.SeqLen != nullptr);
  CHECK(param_.PadSeqLen != nullptr);
  if (param_.Mask->dims().size() == 3) {
    CHECK_EQ(param_.Mask->dims()[2], 1);
  } else {
    CHECK_EQ(param_.Mask->dims().size(), 2);
  }
  return true;
}

bool CalculateLodOp::InferShapeImpl() const {
  auto mask_dims = param_.Mask->dims();
  param_.SeqLod->Resize({mask_dims[0] + 1});
  param_.SeqLen->Resize({mask_dims[0]});
  param_.PadSeqLen->Resize({1});
  return true;
}

bool CalculateLodOp::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  param_.Mask =
      &scope->FindVar(op_desc.Input("Mask").front())->Get<lite::Tensor>();
  param_.SeqLod =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("SeqLod").front());
  param_.SeqLen =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("SeqLen").front());
  param_.PadSeqLen =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("PadSeqLen").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(calculate_lod, paddle::lite::operators::CalculateLodOp);
