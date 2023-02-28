// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/set_value_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SetValueOp::CheckShape() const {
  CHECK(param_.Input);
  CHECK(param_.Out);
  return true;
}

bool SetValueOp::InferShapeImpl() const {
  param_.Out->Resize(param_.Input->dims());
  return true;
}

bool SetValueOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  // Input
  auto input_name = opdesc.Input("Input").front();
  param_.Input = GetTensor(scope, input_name);

  // ValueTensor
  if (opdesc.HasInput("ValueTensor") && !opdesc.Input("ValueTensor").empty()) {
    auto value_tensor_name = opdesc.Input("ValueTensor").front();
    param_.ValueTensor = GetTensor(scope, value_tensor_name);
  }
  // Out
  auto out_name = opdesc.Output("Out").front();
  param_.Out = GetMutableTensor(scope, out_name);
  // StartsTensorList
  param_.StartsTensorList.clear();
  if (opdesc.HasInput("StartsTensorList") &&
      !opdesc.Input("StartsTensorList").empty()) {
    for (auto starts_tensor_name : opdesc.Input("StartsTensorList")) {
      auto starts_tensor = GetTensor(scope, starts_tensor_name);
      CHECK_EQ(starts_tensor->numel(), 1)
          << "The shape of the starts_tensor in must be [1] "
             "in starts_tensor_list, but get "
          << starts_tensor->numel();
      param_.StartsTensorList.push_back(starts_tensor);
    }
  }
  // EndsTensorList
  param_.EndsTensorList.clear();
  if (opdesc.HasInput("EndsTensorList") &&
      !opdesc.Input("EndsTensorList").empty()) {
    for (auto ends_tensor_name : opdesc.Input("EndsTensorList")) {
      auto ends_tensor = GetTensor(scope, ends_tensor_name);
      CHECK_EQ(ends_tensor->numel(), 1)
          << "The shape of the ends_tensor in must be [1] in ends_tensor_list, "
             "but get "
          << ends_tensor->numel();
      param_.EndsTensorList.push_back(ends_tensor);
    }
  }
  // StepsTensorList
  param_.StepsTensorList.clear();
  if (opdesc.HasInput("StepsTensorList") &&
      !opdesc.Input("StepsTensorList").empty()) {
    for (auto steps_tensor_name : opdesc.Input("StepsTensorList")) {
      auto steps_tensor = GetTensor(scope, steps_tensor_name);
      CHECK_EQ(steps_tensor->numel(), 1)
          << "The shape of the steps_tensor in must be [1] in "
             "steps_tensor_list, but get "
          << steps_tensor->numel();
      param_.StepsTensorList.push_back(steps_tensor);
    }
  }
  // Starts
  if (opdesc.HasAttr("starts")) {
    param_.starts = opdesc.GetAttr<std::vector<int64_t>>("starts");
  }
  // Ends
  if (opdesc.HasAttr("ends")) {
    param_.ends = opdesc.GetAttr<std::vector<int64_t>>("ends");
  }
  // Steps
  if (opdesc.HasAttr("steps")) {
    param_.steps = opdesc.GetAttr<std::vector<int64_t>>("steps");
  }
  // Decrease_axes
  if (opdesc.HasAttr("decrease_axes")) {
    param_.decrease_axes =
        opdesc.GetAttr<std::vector<int64_t>>("decrease_axes");
  }
  // None_axes
  if (opdesc.HasAttr("none_axes")) {
    param_.none_axes = opdesc.GetAttr<std::vector<int64_t>>("none_axes");
  }
  // Dtype
  param_.dtype = opdesc.GetAttr<int>("dtype");
  // Axes
  param_.axes = opdesc.GetAttr<std::vector<int64_t>>("axes");
  // Bool_values
  if (opdesc.HasAttr("bool_values")) {
    param_.bool_values = opdesc.GetAttr<std::vector<int>>("bool_values");
  }
  // Fp32_values
  if (opdesc.HasAttr("fp32_values")) {
    param_.fp32_values = opdesc.GetAttr<std::vector<float>>("fp32_values");
  }
  // Int32_values
  if (opdesc.HasAttr("int32_values")) {
    param_.int32_values = opdesc.GetAttr<std::vector<int>>("int32_values");
  }
  // Int64_values
  if (opdesc.HasAttr("int64_values")) {
    param_.int64_values = opdesc.GetAttr<std::vector<int64_t>>("int64_values");
  }
  // Fp64_values
  if (opdesc.HasAttr("fp64_values")) {
    param_.fp64_values = opdesc.GetAttr<std::vector<double>>("fp64_values");
  }
  // Fp16_values
  if (opdesc.HasAttr("fp16_values")) {
    param_.fp16_values = opdesc.GetAttr<std::vector<float>>("fp16_values");
  }
  // Shape
  param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(set_value, paddle::lite::operators::SetValueOp);
