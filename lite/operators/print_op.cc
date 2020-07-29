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

#include "lite/operators/print_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool PrintOp::CheckShape() const {
  CHECK_OR_FALSE(param_.in);
  CHECK_OR_FALSE(param_.out);
  return true;
}

bool PrintOp::InferShapeImpl() const {
  param_.out->set_lod(param_.in->lod());
  param_.out->Resize(param_.in->dims());
  return true;
}

bool PrintOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  AttachParam(&param_);

  param_.name = op_desc.Input("In").front();
  param_.in = scope->FindTensor(param_.name);
  param_.out = scope->FindMutableTensor(op_desc.Output("Out").front());
  param_.first_n = op_desc.GetAttr<int32_t>("first_n");
  param_.message = op_desc.GetAttr<std::string>("message");
  param_.summarize = op_desc.GetAttr<int32_t>("summarize");
  param_.print_tensor_name = op_desc.GetAttr<bool>("print_tensor_name");
  param_.print_tensor_type = op_desc.GetAttr<bool>("print_tensor_type");
  param_.print_tensor_shape = op_desc.GetAttr<bool>("print_tensor_shape");
  param_.print_tensor_lod = op_desc.GetAttr<bool>("print_tensor_lod");
  param_.print_tensor_layout = op_desc.GetAttr<bool>("print_tensor_layout");
  param_.print_phase = op_desc.GetAttr<std::string>("print_phase");
  param_.is_forward = op_desc.GetAttr<bool>("is_forward");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(print, paddle::lite::operators::PrintOp);
