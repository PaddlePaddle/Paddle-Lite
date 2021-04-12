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

#include "lite/operators/io_copy_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool IoCopyOp::CheckShape() const {
  CHECK(param_.x != nullptr || param_.x_array != nullptr);
  if (param_.x != nullptr) {
    CHECK(param_.y != nullptr);
  }
  if (param_.x_array != nullptr) {
    CHECK(param_.y_array != nullptr);
  }
  return true;
}

bool IoCopyOp::InferShapeImpl() const {
  if (param_.x != nullptr) {
    param_.y->Resize(param_.x->dims());
    param_.y->set_lod(param_.x->lod());
    param_.y->set_precision(param_.x->precision());
    param_.y->set_persistable(param_.x->persistable());
  }
  if (param_.x_array != nullptr) {
    param_.y_array->resize(param_.x_array->size());
    for (size_t i = 0; i < param_.x_array->size(); i++) {
      param_.y_array->at(i).Resize(param_.x_array->at(i).dims());
      param_.y_array->at(i).set_lod(param_.x_array->at(i).lod());
      param_.y_array->at(i).set_precision(param_.x_array->at(i).precision());
      param_.y_array->at(i).set_persistable(
          param_.x_array->at(i).persistable());
    }
  }
  return true;
}

bool IoCopyOp::Run() { return OpLite::Run(); }

bool IoCopyOp::AttachImpl(const cpp::OpDesc &opdesc,
                          paddle::lite::Scope *scope) {
  if (opdesc.HasInput("Input")) {
    param_.x = scope->FindTensor(opdesc.Input("Input").front());
  }
  if (opdesc.HasInput("InputArray")) {
    param_.x_array = scope->FindTensorList(opdesc.Input("InputArray").front());
  }
  if (opdesc.HasOutput("Out")) {
    param_.y = scope->FindMutableTensor(opdesc.Output("Out").front());
  }
  if (opdesc.HasOutput("OutArray")) {
    param_.y_array =
        scope->FindMutableTensorList(opdesc.Output("OutArray").front());
  }
  if (opdesc.HasAttr("process_type")) {
    param_.process_type = opdesc.GetAttr<int>("process_type");
  }
  return true;
}

std::string IoCopyOp::DebugString() const { return "io_copy_op"; }

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(io_copy, paddle::lite::operators::IoCopyOp);
