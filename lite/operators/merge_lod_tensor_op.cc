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

#include "lite/operators/merge_lod_tensor_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MergeLodTensorOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.mask);
  CHECK_OR_FALSE(param_.in_true);
  CHECK_OR_FALSE(param_.in_false);
  CHECK_OR_FALSE(param_.out);

  const auto mask_dims = param_.mask->dims();
  CHECK_OR_FALSE(mask_dims.size() == 2);
  CHECK_OR_FALSE(mask_dims[1] == 1);

  return true;
}

bool MergeLodTensorOpLite::InferShapeImpl() const {
  auto dims = param_.in_true->dims();
  param_.out->Resize(dims);
  return true;
}

bool MergeLodTensorOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                      lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto mask = op_desc.Input("Mask").front();
  auto in_true = op_desc.Input("InTrue").front();
  auto in_false = op_desc.Input("InFalse").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.mask = scope->FindVar(mask)->GetMutable<lite::Tensor>();
  param_.in_true = scope->FindVar(in_true)->GetMutable<lite::Tensor>();
  param_.in_false = scope->FindVar(in_false)->GetMutable<lite::Tensor>();

  auto out = op_desc.Output("Out").front();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  param_.level = op_desc.GetAttr<int>("level");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(merge_lod_tensor,
                 paddle::lite::operators::MergeLodTensorOpLite);
