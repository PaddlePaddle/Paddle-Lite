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

#include "lite/operators/expand_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ExpandOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);

  int x_dims_size = param_.X->dims().size();
  CHECK_LE(x_dims_size, 6)
      << "The rank of Input(X) must not be greater than 6.";

  int expand_size = 0;
  if (param_.ExpandTimes != nullptr) {
    expand_size = param_.ExpandTimes->numel();
  } else if (!param_.expand_times_tensor.empty()) {
    expand_size = param_.expand_times_tensor.size();
  } else {
    expand_size = param_.expand_times.size();
  }
  CHECK_EQ(expand_size, x_dims_size)
      << "The number of expand_times size must be qual to the rank of "
         "Input(X).";

  return true;
}

bool ExpandOpLite::InferShapeImpl() const {
  std::vector<int> expand_times;
  if (param_.ExpandTimes != nullptr) {
    auto expand_times_data = param_.ExpandTimes->template data<int>();
    for (int64_t i = 0; i < param_.ExpandTimes->numel(); i++) {
      expand_times.push_back(expand_times_data[i]);
    }
  } else if (!param_.expand_times_tensor.empty()) {
    for (size_t i = 0; i < param_.expand_times_tensor.size(); i++) {
      expand_times.push_back(
          param_.expand_times_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_times = param_.expand_times;
  }

  DDim out_dims(param_.X->dims());
  for (size_t i = 0; i < expand_times.size(); ++i) {
    out_dims[i] *= static_cast<int64_t>(expand_times[i]);
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool ExpandOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);

  if (opdesc.HasInput("ExpandTimes") && !opdesc.Input("ExpandTimes").empty()) {
    auto expand_times_tensor_name = opdesc.Input("ExpandTimes").front();
    param_.ExpandTimes =
        GetMutableVar<lite::Tensor>(scope, expand_times_tensor_name);
  }
  param_.expand_times_tensor.clear();  // Avoid errors caused by repeated calls
  if (opdesc.HasInput("expand_times_tensor") &&
      !opdesc.Input("expand_times_tensor").empty()) {
    for (auto expand_times_tensor_name : opdesc.Input("expand_times_tensor")) {
      param_.expand_times_tensor.push_back(
          GetMutableVar<lite::Tensor>(scope, expand_times_tensor_name));
    }
  }

  param_.expand_times = opdesc.GetAttr<std::vector<int>>("expand_times");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(expand, paddle::lite::operators::ExpandOpLite);
