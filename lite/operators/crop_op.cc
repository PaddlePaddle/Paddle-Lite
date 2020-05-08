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

#include "lite/operators/crop_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CropOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool CropOpLite::InferShapeImpl() const {
  // nchw
  auto x_dims = param_.X->dims();
  lite::DDim output_shape(x_dims);
  output_shape[0] = x_dims[0];
  output_shape[1] = param_.shape[1];
  output_shape[2] = param_.shape[2];
  output_shape[3] = param_.shape[3];
  param_.Out->Resize(output_shape);
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool CropOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.offsets = op_desc.GetAttr<std::vector<int>>("offsets");
  param_.shape = op_desc.GetAttr<std::vector<int>>("shape");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(crop, paddle::lite::operators::CropOpLite);
