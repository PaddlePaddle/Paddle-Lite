// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/crop_tensor_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CropTensorOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool CropTensorOpLite::InferShapeImpl() const {
  std::vector<int64_t> shape;
  if (param_.Shape != nullptr) {
    auto shape_data = param_.Shape->template data<int>();
    for (int64_t i = 0; i < param_.Shape->numel(); i++) {
      shape.push_back(shape_data[i]);
    }
  } else if (param_.ShapeTensor != nullptr) {
    for (size_t i = 0; i < param_.ShapeTensor->size(); i++) {
      shape.push_back(param_.ShapeTensor->at(i).template data<int>()[0]);
    }
  } else {
    shape = std::vector<int64_t>(param_.shape.begin(), param_.shape.end());
  }
  param_.Out->Resize(shape);
  return true;
}

bool CropTensorOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                  lite::Scope *scope) {
  param_.X = scope->FindTensor(op_desc.Input("X").front());
  param_.Out = scope->FindMutableTensor(op_desc.Output("Out").front());

  if (op_desc.HasInput("Shape")) {
    auto names = op_desc.Input("Shape");
    if (!names.empty()) {
      param_.Shape = scope->FindTensor(names.front());
    }
  }
  if (op_desc.HasInput("Offsets")) {
    auto names = op_desc.Input("Offsets");
    if (!names.empty()) {
      param_.Offsets = scope->FindTensor(names.front());
    }
  }
  if (op_desc.HasInput("ShapeTensor")) {
    auto names = op_desc.Input("ShapeTensor");
    if (!names.empty()) {
      param_.ShapeTensor = scope->FindTensorList(names.front());
    }
  }
  if (op_desc.HasInput("OffsetsTensor")) {
    auto names = op_desc.Input("OffsetsTensor");
    if (!names.empty()) {
      param_.OffsetsTensor = scope->FindTensorList(names.front());
    }
  }

  param_.offsets = op_desc.GetAttr<std::vector<int>>("offsets");
  param_.shape = op_desc.GetAttr<std::vector<int>>("shape");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(crop_tensor, paddle::lite::operators::CropTensorOpLite);
