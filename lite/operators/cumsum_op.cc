// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/cumsum_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CumsumOpLite::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Out);
  auto x_rank = param_.X->dims().size();
  CHECK(param_.axis >= -static_cast<int>(x_rank) &&
        param_.axis < static_cast<int>(x_rank))
      << "axis: " << param_.axis << ", x_dims: " << param_.X->dims();
  return true;
}

bool CumsumOpLite::InferShapeImpl() const {
  if (param_.flatten) {
    param_.Out->Resize(DDim{{param_.X->numel()}});
  } else {
    param_.Out->Resize(param_.X->dims());
  }
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool CumsumOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindTensor(op_desc.Input("X").front());
  param_.Out = scope->FindMutableTensor(op_desc.Output("Out").front());

  param_.axis = op_desc.GetAttr<int>("axis");
  param_.exclusive = op_desc.GetAttr<bool>("exclusive");
  param_.reverse = op_desc.GetAttr<bool>("reverse");
  if (op_desc.HasAttr("flatten")) {
    param_.flatten = op_desc.GetAttr<bool>("flatten");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(cumsum, paddle::lite::operators::CumsumOpLite);
