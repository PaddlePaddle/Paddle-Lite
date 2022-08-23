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

#include "lite/operators/calib_inplace_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CalibInplaceOpLite::CheckShape() const { return true; }
bool CalibInplaceOpLite::InferShapeImpl() const { return true; }
#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
bool CalibInplaceOpLite::AttachImpl(const cpp::OpDescWrite &opdesc,
                                    lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("Input").front());
  CHECK(x_var);
  param_.input = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.output = param_.input;
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (opdesc.HasAttr("scale")) {
    param_.scale = opdesc.GetAttr<float>("scale");
  }
  CHECK(param_.input) << "Input(X) of CalibInplaceOp should not be null.";
  CHECK(param_.output) << "Output(Out) of CalibInplaceOp should not be null.";
  return true;
}
#endif
bool CalibInplaceOpLite::AttachImpl(const cpp::OpDesc &opdesc,
                                    lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("Input").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.input = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.output = param_.input;
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (opdesc.HasAttr("scale")) {
    param_.scale = opdesc.GetAttr<float>("scale");
  }
  CHECK(param_.input) << "Input(X) of CalibInplaceOp should not be null.";
  CHECK(param_.output) << "Output(Out) of CalibInplaceOp should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(calib_inplace, paddle::lite::operators::CalibInplaceOpLite);
