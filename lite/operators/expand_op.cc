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
  CHECK_LE(x_dims_size, 6u)
      << "The rank of Input(X) must not be greater than 6.";

  int expand_size = 0;
  if (param_.ExpandTimes != nullptr) {
    expand_size = param_.ExpandTimes->numel();
  } else if (param_.expand_times_tensor != nullptr) {
    expand_size = param_.expand_times_tensor->size();
  } else {
    expand_size = param_.expand_times.size();
  }
  CHECK_EQ(expand_size, x_dims_size)
      << "The number of expand_times size must be qual to the rank of "
         "Input(X).";

  return true;
}

bool ExpandOpLite::InferShapeImpl() const {
  DDim out_dims(param_.X->dims());
  for (size_t i = 0; i < param_.expand_times.size(); ++i) {
    out_dims[i] *= param_.expand_times[i];
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool ExpandOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);

  if (opdesc.HasInput("ExpandTimes")) {
    param_.ExpandTimes = scope->FindTensor("ExpandTimes");
  }
  if (opdesc.HasInput("expand_times_tensor")) {
    param_.expand_times_tensor = scope->FindTensorList("expand_times_tensor");
  }

  param_.expand_times = opdesc.GetAttr<std::vector<int>>("expand_times");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(expand, paddle::lite::operators::ExpandOpLite);
