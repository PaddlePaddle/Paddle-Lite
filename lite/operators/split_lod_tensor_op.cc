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

#include "lite/operators/split_lod_tensor_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SplitLodTensorOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.mask);
  CHECK_OR_FALSE(param_.out_true);
  CHECK_OR_FALSE(param_.out_false);

  const auto mask_dims = param_.mask->dims();
  CHECK_OR_FALSE(mask_dims.size() == 2);
  CHECK_OR_FALSE(mask_dims[1] == 1);

  return true;
}

bool SplitLodTensorOpLite::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  param_.out_true->Resize(x_dims);
  param_.out_false->Resize(x_dims);
  return true;
}

bool SplitLodTensorOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                      lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto mask = op_desc.Input("Mask").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.mask = scope->FindVar(mask)->GetMutable<lite::Tensor>();

  auto out_true = op_desc.Output("OutTrue").front();
  auto out_false = op_desc.Output("OutFalse").front();
  param_.out_true = scope->FindVar(out_true)->GetMutable<lite::Tensor>();
  param_.out_false = scope->FindVar(out_false)->GetMutable<lite::Tensor>();

  param_.level = op_desc.GetAttr<int>("level");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(split_lod_tensor,
                 paddle::lite::operators::SplitLodTensorOpLite);
