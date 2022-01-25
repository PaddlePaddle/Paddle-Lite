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

#include "lite/operators/gaussian_random_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool GaussRandomOp::CheckShape() const {
  if (param_.ShapeTensor == nullptr && param_.ShapeTensorList.empty()) {
    CHECK(param_.shape.size() > 0)
        << "Attribute(shape) of GaussRandomOp must be set and shape.size() > 0";
  }
  return true;
}

bool GaussRandomOp::InferShapeImpl() const {
  auto shape = param_.shape;
  std::vector<int64_t> temp{};
  temp.reserve(shape.size());
  for (auto dim : shape) {
    temp.push_back(static_cast<int64_t>(dim));
  }
  if (shape.empty() && param_.ShapeTensor != nullptr) {
    auto shape_dims = param_.ShapeTensor->dims();
    int num_ele = 1;
    for (int i = 0; i < shape_dims.size(); ++i) {
      num_ele *= shape_dims[i];
    }
    auto vec_dims = std::vector<int64_t>(num_ele, 1);
    DDimLite dims(vec_dims);
    param_.Out->Resize(dims);
    return true;
  }
  DDimLite dims(temp);
  param_.Out->Resize(dims);
  return true;
}

bool GaussRandomOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto Out = op_desc.Output("Out").front();
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  if (op_desc.HasInput("ShapeTensor")) {
    auto x = op_desc.Input("ShapeTensor");
    if (x.size() > 0)
      param_.ShapeTensor =
          scope->FindVar(x.front())->GetMutable<lite::Tensor>();
    else
      param_.ShapeTensor = nullptr;
  }
  if (op_desc.HasInput("ShapeTensorList")) {
    param_.ShapeTensorList.clear();
    auto stlist = op_desc.Input("ShapeTensorList");
    if (stlist.size() > 0) {
      for (auto var : stlist) {
        param_.ShapeTensorList.push_back(
            scope->FindVar(var)->GetMutable<lite::Tensor>());
      }
    }
  }
  if (op_desc.HasAttr("mean")) {
    param_.mean = op_desc.GetAttr<float>("mean");
  }
  if (op_desc.HasAttr("seed")) {
    param_.seed = op_desc.GetAttr<int>("seed");
  }
  if (op_desc.HasAttr("dtype")) {
    param_.dtype = op_desc.GetAttr<int>("dtype");
  }
  if (op_desc.HasAttr("shape")) {
    param_.shape = op_desc.GetAttr<std::vector<int64_t>>("shape");
  }
  if (op_desc.HasAttr("std")) {
    param_.gauss_std = op_desc.GetAttr<float>("std");
  }
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(gaussian_random, paddle::lite::operators::GaussRandomOp);
