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

#include "lite/operators/box_coder_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BoxCoderOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.prior_box);
  CHECK_OR_FALSE(param_.prior_box_var);
  CHECK_OR_FALSE(param_.target_box);
  CHECK_OR_FALSE(param_.proposals);
  return true;
}

bool BoxCoderOpLite::InferShape() const {
  param_.proposals->Resize(param_.target_box->dims());
  return true;
}

bool BoxCoderOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  LOG(INFO) << "Attach Impl succeed!";
  auto Prior_box_name = opdesc.Input("PriorBox").front();
  auto Prior_box_var_name = opdesc.Input("PriorBoxVar").front();
  auto Target_box_name = opdesc.Input("TargetBox").front();
  auto Output_box_name = opdesc.Output("OutputBox").front();

  param_.prior_box = GetVar<lite::Tensor>(scope, Prior_box_name);
  param_.prior_box_var = GetVar<lite::Tensor>(scope, Prior_box_var_name);
  param_.target_box = GetVar<lite::Tensor>(scope, Target_box_name);
  param_.proposals = GetMutableVar<lite::Tensor>(scope, Output_box_name);
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  }
  param_.box_normalized = opdesc.GetAttr<bool>("box_normalized");
  param_.code_type = opdesc.GetAttr<std::string>("code_type");
  LOG(INFO) << "Attach Impl exit!";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(box_coder, paddle::lite::operators::BoxCoderOpLite);
