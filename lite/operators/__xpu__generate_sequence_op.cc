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

#include "lite/operators/__xpu__generate_sequence_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUGenerateSequenceOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.input->dims();
  auto x_rank = x_dims.size();
  CHECK(param_.axis >= -static_cast<int>(x_rank) &&
        param_.axis < static_cast<int>(x_rank))
      << "axis: " << param_.axis << ", x_dims: " << x_dims;

  return true;
}

bool XPUGenerateSequenceOp::InferShapeImpl() const {
  auto out = param_.output;
  if (param_.flatten) {
    out->Resize(DDim{{param_.input->numel()}});
  } else {
    out->Resize(param_.input->dims());
  }
  out->set_lod(param_.input->lod());
  return true;
}

bool XPUGenerateSequenceOp::AttachImpl(const cpp::OpDesc &opdesc,
                                       lite::Scope *scope) {
  param_.input = scope->FindTensor(opdesc.Input("X").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.flatten = opdesc.GetAttr<bool>("flatten");
  param_.value = opdesc.GetAttr<float>("value");
  param_.dtype = opdesc.GetAttr<int>("dtype");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__generate_sequence,
                 paddle::lite::operators::XPUGenerateSequenceOp);
