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

#include "lite/operators/attention_padding_mask_op.h"
#include "lite/core/op_registry.h"
#include "lite/core/scope.h"

namespace paddle {
namespace lite {
namespace operators {

bool AttentionPaddingMaskOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.pad_begin);
  return true;
}

bool AttentionPaddingMaskOp::InferShapeImpl() const {
  auto src_len = param_.X->lod()[0][1];
  CHECK_EQ(src_len, param_.X->dims()[1])
      << "Mismatch source length, expect: " << src_len
      << ", get: " << param_.X->lod()[0][1];
  auto att_batch = param_.X->lod()[0].size() - 1;
  auto src_batch = param_.Y->lod()[0].size() - 1;
  CHECK_EQ(att_batch % src_batch, 0)
      << "Mismatch batch size, bottom0: " << att_batch
      << ", bottom1: " << src_batch;

  param_.pad_begin->Resize({static_cast<int64_t>(src_batch)});
  param_.Out->Resize(param_.X->dims());
  param_.Out->set_lod(param_.X->lod());

  return true;
}

bool AttentionPaddingMaskOp::AttachImpl(const cpp::OpDesc &op_desc,
                                        lite::Scope *scope) {
  param_.X = scope->FindTensor(op_desc.Input("X").front());
  param_.Y = scope->FindTensor(op_desc.Input("Y").front());
  param_.Out = scope->FindMutableTensor(op_desc.Output("Out").front());
  param_.pad_begin =
      scope->FindMutableTensor(op_desc.Output("pad_begin").front());

  param_.pad_id = op_desc.GetAttr<int>("pad_id");
  param_.mask = op_desc.GetAttr<float>("mask");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(attention_padding_mask,
                 paddle::lite::operators::AttentionPaddingMaskOp);
REGISTER_LITE_OP(search_attention_padding_mask,
                 paddle::lite::operators::AttentionPaddingMaskOp);
