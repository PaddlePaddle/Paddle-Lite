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

#include "lite/operators/uniform_random_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UniformRandomOpLite::CheckShape() const { return true; }

bool UniformRandomOpLite::InferShapeImpl() const {
  std::vector<int64_t> out_shape;
  auto* shape_tensor = param_.shape_tensor;
  auto& shape_tensor_list = param_.shape_tensor_list;
  if (shape_tensor) {
    if (shape_tensor->precision() == PrecisionType::kInt64) {
      auto* shape_tensor_data = shape_tensor->data<int64_t>();
      for (int i = 0; i < shape_tensor->numel(); i++) {
        out_shape.push_back(shape_tensor_data[i]);
      }
    } else if (shape_tensor->precision() == PrecisionType::kInt32) {
      auto* shape_tensor_data = shape_tensor->data<int32_t>();
      for (int i = 0; i < shape_tensor->numel(); i++) {
        out_shape.push_back(shape_tensor_data[i]);
      }
    } else {
      LOG(ERROR) << "The dtype of shape tensor must be int32 or int64.";
    }
  } else if (!shape_tensor_list.empty()) {
    for (size_t i = 0; i < shape_tensor_list.size(); i++) {
      auto* shape_tensor = shape_tensor_list[i];
      if (shape_tensor->precision() == PrecisionType::kInt64) {
        out_shape.push_back(shape_tensor->data<int64_t>()[0]);
      } else if (shape_tensor->precision() == PrecisionType::kInt32) {
        out_shape.push_back(shape_tensor->data<int32_t>()[0]);
      } else {
        LOG(ERROR) << "The dtype of shape tensor must be int32 or int64.";
      }
    }
  } else if (!param_.shape.empty()) {
    out_shape = param_.shape;
  } else {
    LOG(FATAL) << "no valid out_shape. Must set one of shape_tensor, or "
                  "shape_tensor_list, or shape.";
  }

  param_.Out->Resize(out_shape);
  return true;
}

bool UniformRandomOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                     lite::Scope* scope) {
  param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
  param_.min = opdesc.GetAttr<float>("min");
  param_.max = opdesc.GetAttr<float>("max");
  param_.seed = opdesc.GetAttr<int>("seed");
  param_.dtype = opdesc.GetAttr<int>("dtype");
  param_.shape_tensor = nullptr;
  if (opdesc.HasInput("ShapeTensor") && !opdesc.Input("ShapeTensor").empty()) {
    auto shape_tensor_name = opdesc.Input("ShapeTensor").front();
    param_.shape_tensor = GetMutableVar<lite::Tensor>(scope, shape_tensor_name);
  }
  param_.shape_tensor_list.clear();  // Avoid errors caused by repeated calls
  if (opdesc.HasInput("ShapeTensorList") &&
      !opdesc.Input("ShapeTensorList").empty()) {
    for (auto shape_tensor_name : opdesc.Input("ShapeTensorList")) {
      param_.shape_tensor_list.push_back(
          GetMutableVar<lite::Tensor>(scope, shape_tensor_name));
    }
  }
  param_.Out = GetMutableVar<Tensor>(scope, opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(uniform_random, paddle::lite::operators::UniformRandomOpLite);
