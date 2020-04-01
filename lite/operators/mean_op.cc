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

#include "lite/operators/mean_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MeanOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool MeanOp::InferShapeImpl() const {
  param_.Out->Resize(std::vector<int64_t>{1});
  return true;
}

bool MeanOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<Tensor>(scope, Out_name);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(mean, paddle::lite::operators::MeanOp);
