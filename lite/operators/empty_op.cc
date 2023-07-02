// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/empty_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool EmptyOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool EmptyOp::InferShapeImpl() const {
  std::vector<int64_t> OutShape;
  auto ShapeTensor = param_.ShapeTensor;
  auto ShapeTensorList = param_.ShapeTensorList;
  if (ShapeTensor != nullptr) {
    auto ShapeTensorData = ShapeTensor->data<int>();
    for (int i = 0; i < ShapeTensor->numel(); i++) {
      OutShape.push_back(ShapeTensorData[i]);
    }
  } else if (!ShapeTensorList.empty()) {
    for (size_t i = 0; i < ShapeTensorList.size(); i++) {
      OutShape.push_back(ShapeTensorList[i]->data<int>()[0]);
    }
  } else if (!param_.shape.empty()) {
    OutShape = param_.shape;
  } else {
    LOG(WARNING) << "EmptyOp output is 0D-tensor.";
  }

  param_.Out->Resize(OutShape);
  return true;
}

bool EmptyOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  if (opdesc.HasInput("ShapeTensor") && !opdesc.Input("ShapeTensor").empty()) {
    param_.ShapeTensor =
        scope->FindMutableTensor(opdesc.Input("ShapeTensor").front());
  }
  param_.ShapeTensorList.clear();
  if (opdesc.HasInput("ShapeTensorList") &&
      !opdesc.Input("ShapeTensorList").empty()) {
    for (auto name : opdesc.Input("ShapeTensorList")) {
      param_.ShapeTensorList.push_back(
          GetMutableVar<lite::Tensor>(scope, name));
    }
  }
  if (opdesc.HasAttr("shape")) {
    auto type = opdesc.GetAttrType("shape");
    if (type == OpAttrType::INTS) {  // paddle1.0 shape type is ints
      auto shape = opdesc.GetAttr<std::vector<int32_t>>("shape");
      param_.shape.resize(shape.size());
      for (int i = 0; i < shape.size(); i++) {
        param_.shape[i] = shape[i];
      }
    } else {
      param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
    }
  }
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  CHECK(param_.Out) << "Output(Out) of EmptyOp should not be null.";
  if (opdesc.HasAttr("dtype")) {
    param_.dtype = opdesc.GetAttr<int>("dtype");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(empty, paddle::lite::operators::EmptyOp);
