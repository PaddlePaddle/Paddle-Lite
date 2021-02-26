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
  std::vector<int64_t> shape;
  if (param_.Y != nullptr) {
    auto shape_data = param_.Y->template data<int>();
    for (int64_t i = 0; i < param_.Y->numel(); i++) {
      shape.push_back(shape_data[i]);
    }
  } else {
    shape = std::vector<int64_t>(param_.shape.begin(), param_.shape.end());
    if (shape[0] == -1) {
      auto x_dims = param_.X->dims();
      shape[0] = x_dims[0];
    }
  }
  param_.Out->Resize(shape);
  return true;
}

bool CropOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindTensor(op_desc.Input("X").front());
  if (op_desc.HasInput("Y")) {
    auto names = op_desc.Input("Y");
    if (!names.empty()) {
      param_.Y = scope->FindTensor(names.front());
    }
  }
  if (op_desc.HasInput("Offsets")) {
    auto names = op_desc.Input("Offsets");
    if (!names.empty()) {
      param_.Offsets = scope->FindTensor(names.front());
    }
  }
  param_.Out = scope->FindMutableTensor(op_desc.Output("Out").front());

  param_.offsets = op_desc.GetAttr<std::vector<int>>("offsets");
  param_.shape = op_desc.GetAttr<std::vector<int>>("shape");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(crop, paddle::lite::operators::CropOpLite);
