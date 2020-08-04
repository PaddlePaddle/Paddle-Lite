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

#include "lite/operators/expand_as_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ExpandAsOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Target);
  CHECK_OR_FALSE(param_.Out);
  int target_size = param_.Target->dims().size();
  int x_dims_size = param_.X->dims().size();
  CHECK_EQ(target_size, x_dims_size)
      << "The number of expand_times size must be qual to the rank of "
         "Input(X).";
  CHECK_LE(param_.X->dims().size(), 6u)
      << "The rank of Input(X) must not be greater than 6.";
  return true;
}

bool ExpandAsOpLite::InferShapeImpl() const {
  DDim out_dims(param_.X->dims());
  for (size_t i = 0; i < param_.Target->dims().size(); ++i) {
    // out_dims[i] *= param_.expand_times[i];
    out_dims[i] = param_.Target->dims()[i];
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool ExpandAsOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  auto Target_name = opdesc.Input("Target").front();
  param_.Target = GetVar<lite::Tensor>(scope, Target_name);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(expand_as, paddle::lite::operators::ExpandAsOpLite);
