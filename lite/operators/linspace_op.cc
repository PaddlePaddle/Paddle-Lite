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

#include "lite/operators/linspace_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LinspaceOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Start);
  CHECK_OR_FALSE(param_.Stop);
  CHECK_OR_FALSE(param_.Num);
  CHECK_OR_FALSE(param_.Out);

  int start_dims_size = param_.Start->dims().size();
  CHECK_EQ(start_dims_size, 1) << "The shape of input start must be 1.";
  int stop_dims_size = param_.Stop->dims().size();
  CHECK_EQ(stop_dims_size, 1) << "The shape of input stop must be 1.";
  int num_dims_size = param_.Num->dims().size();
  CHECK_EQ(num_dims_size, 1) << "The shape of input num must be 1.";

  return true;
}

bool LinspaceOpLite::InferShapeImpl() const {
  // param_.dtype(int) is defined in paddle/fluid/framework/framework.proto
  // param_.dtype(int) means output dtype and lite supports fp32/int32.
  // if param_.dtype is not defined, output dtype is fp32.
  switch (param_.dtype) {
    case 2:
      param_.Out->set_precision(PRECISION(kInt32));
      break;
    case 5:
      param_.Out->set_precision(PRECISION(kFloat));
      break;
    default:
      param_.Out->set_precision(PRECISION(kFloat));
      break;
  }
  param_.Out->Resize(std::vector<int64_t>{param_.Num->data<int32_t>()[0]});
  return true;
}

bool LinspaceOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto start_name = opdesc.Input("Start").front();
  auto stop_name = opdesc.Input("Stop").front();
  auto num_name = opdesc.Input("Num").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.Start = GetVar<lite::Tensor>(scope, start_name);
  param_.Stop = GetVar<lite::Tensor>(scope, stop_name);
  param_.Num = GetVar<lite::Tensor>(scope, num_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);

  if (opdesc.HasAttr("dtype")) {
    param_.dtype = opdesc.GetAttr<int>("dtype");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(linspace, paddle::lite::operators::LinspaceOpLite);
