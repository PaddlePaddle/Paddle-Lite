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

#include "lite/operators/__xpu__resnet50_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUResNet50Op::CheckShape() const { return true; }

bool XPUResNet50Op::InferShapeImpl() const {
  auto input_shape = param_.input->dims();
  input_shape[1] = 2048;
  input_shape[2] = 1;
  input_shape[3] = 1;
  param_.output->Resize(input_shape);
  return true;
}

bool XPUResNet50Op::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.output = scope->FindVar(op_desc.Output("Output").front())
                      ->GetMutable<lite::Tensor>();

  param_.filter.clear();
  for (auto& name : op_desc.Input("Filter")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.filter.push_back(t);
  }
  param_.bias.clear();
  for (auto& name : op_desc.Input("Bias")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.bias.push_back(t);
  }
  param_.max_filter.clear();
  for (auto& name : op_desc.Input("MaxFilter")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.max_filter.push_back(t);
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__resnet50, paddle::lite::operators::XPUResNet50Op);
