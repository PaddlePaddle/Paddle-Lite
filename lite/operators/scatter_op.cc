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

#include "lite/operators/scatter_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ScatterOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ScatterOp::InferShapeImpl() const {
  auto index_dims = param_.indexs->dims();
  auto update_dims = param_.updates->dims();
  auto input_dims = param_.x->dims();
  for (int i = 1; i < update_dims.size(); i++) {
    CHECK_EQ_OR_FALSE(update_dims[i], input_dims[i]);
  }
  CHECK_EQ_OR_FALSE(index_dims.size(), 1L);
  param_.output->Resize(input_dims);
  return true;
}

bool ScatterOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto indexs = op_desc.Input("Ids").front();
  auto updates = op_desc.Input("Updates").front();
  auto output = op_desc.Output("Out").front();
  if (op_desc.HasAttr("overwrite")) {
    param_.overwrite = op_desc.GetAttr<bool>("overwrite");
  } else {
    param_.overwrite = true;
  }
  param_.x = scope->FindVar(x)->GetMutable<Tensor>();
  param_.indexs = scope->FindVar(indexs)->GetMutable<Tensor>();
  param_.updates = scope->FindVar(updates)->GetMutable<Tensor>();
  param_.output = scope->FindMutableTensor(output);

  CHECK(param_.x);
  CHECK(param_.indexs);
  CHECK(param_.updates);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(scatter, paddle::lite::operators::ScatterOp);
