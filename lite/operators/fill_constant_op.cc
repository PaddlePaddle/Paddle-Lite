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

#include "lite/operators/fill_constant_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FillConstantOp::CheckShape() const {
  CHECK(param_.out);
  return true;
}

bool FillConstantOp::InferShapeImpl() const {
  std::vector<int64_t> out_shape;
  auto shape_tensor = param_.shape_tensor;
  auto shape_tensor_list = param_.shape_tensor_list;
  if (shape_tensor != nullptr) {
    auto shape_tensor_data = shape_tensor->data<int>();
    for (int i = 0; i < shape_tensor->numel(); i++) {
      out_shape.push_back(shape_tensor_data[i]);
    }
  } else if (!shape_tensor_list.empty()) {
    for (size_t i = 0; i < shape_tensor_list.size(); i++) {
      out_shape.push_back(shape_tensor_list[i]->data<int>()[0]);
    }
  } else if (!param_.shape.empty()) {
    out_shape = param_.shape;
  } else {
    LOG(FATAL) << "no valid out_shape. Must set one of shape_tensor, or "
                  "shape_tensor_list, or shape.";
  }

  param_.out->Resize(out_shape);
  return true;
}

bool FillConstantOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto out_name = opdesc.Output("Out").front();

  param_.out = GetMutableVar<lite::Tensor>(scope, out_name);
  param_.dtype = opdesc.GetAttr<int>("dtype");
  if (opdesc.HasAttr("shape")) {
    param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
  }
  param_.value = opdesc.GetAttr<float>("value");
  param_.force_cpu = opdesc.GetAttr<bool>("force_cpu");

  if (opdesc.HasInput("ShapeTensor") && !opdesc.Input("ShapeTensor").empty()) {
    auto shape_tensor_name = opdesc.Input("ShapeTensor").front();
    param_.shape_tensor = GetMutableVar<lite::Tensor>(scope, shape_tensor_name);
  }
  if (opdesc.HasInput("ShapeTensorList") &&
      !opdesc.Input("ShapeTensorList").empty()) {
    for (auto shape_tensor_name : opdesc.Input("ShapeTensorList")) {
      param_.shape_tensor_list.push_back(
          GetMutableVar<lite::Tensor>(scope, shape_tensor_name));
    }
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fill_constant, paddle::lite::operators::FillConstantOp);
