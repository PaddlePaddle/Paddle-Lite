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

#include "lite/operators/scale_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ScaleOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ScaleOp::InferShapeImpl() const {
  param_.output->Resize(param_.x->dims());
  return true;
}

bool ScaleOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  param_.x = scope->FindVar(x)->GetMutable<Tensor>();
  param_.output = scope->FindMutableTensor(output);
  param_.scale = op_desc.GetAttr<float>("scale");
  param_.bias = op_desc.GetAttr<float>("bias");
  param_.bias_after_scale = op_desc.GetAttr<bool>("bias_after_scale");
  param_.alpha = 6.f;  // default value for placeholder of element+scale pass
  if (op_desc.HasAttr("activation_type")) {
    auto act_type = op_desc.GetAttr<std::string>("activation_type");
    param_.activation_type = act_type;
    if (act_type == "relu") {
      param_.fuse_relu = true;
    } else if (act_type == "relu6") {
      param_.alpha = op_desc.GetAttr<float>("alpha");  // 6.f
    } else if (act_type == "leaky_relu") {
      param_.alpha = op_desc.GetAttr<float>("alpha");
    } else {
      CHECK(false)
          << "The fused conv only supports fuse with relu and leaky relu";
    }

    if (op_desc.HasAttr("fuse_scaleact")) {
      param_.fuse_scaleact = op_desc.GetAttr<bool>("fuse_scaleact");
      param_.scale1 = op_desc.GetAttr<float>("scale1");
      param_.bias1 = op_desc.GetAttr<float>("bias1");
    }
  }
  CHECK(param_.x);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(scale, paddle::lite::operators::ScaleOp);
