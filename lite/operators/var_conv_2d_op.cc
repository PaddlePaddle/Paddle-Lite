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

#include "lite/operators/var_conv_2d_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool VarConv2dOp::CheckShape() const { return true; }

bool VarConv2dOp::InferShapeImpl() const { return true; }

bool VarConv2dOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  if (opdesc.HasInput("ROW") && !opdesc.Input("ROW").empty()) {
    param_.ROW = const_cast<lite::Tensor *>(
        &scope->FindVar(opdesc.Input("ROW").front())->Get<lite::Tensor>());
    CHECK(param_.ROW) << "Input(ROW) of VarConv2dOP should not be null.";
  }
  if (opdesc.HasInput("COLUMN") && !opdesc.Input("COLUMN").empty()) {
    param_.COLUMN = const_cast<lite::Tensor *>(
        &scope->FindVar(opdesc.Input("COLUMN").front())->Get<lite::Tensor>());
    CHECK(param_.COLUMN) << "Input(COLUMN) of VarConv2dOP should not be null.";
  }
  param_.W = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("W").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.Col =
      scope->FindVar(opdesc.Output("Col").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X) << "X(Input) of VarConv2dOP should not be null.";
  CHECK(param_.W) << "W(Input) of VarConv2dOP should not be null.";
  CHECK(param_.Out) << "Out(Output) of VarConv2dOP should not be null.";
  CHECK(param_.Col) << "Col(Output) of VarConv2dOP should not be null.";
  param_.output_channel = opdesc.GetAttr<int>("OutputChannel");
  param_.input_channel = opdesc.GetAttr<int>("InputChannel");
  param_.kernel_h = opdesc.GetAttr<int>("KernelH");
  param_.kernel_w = opdesc.GetAttr<int>("KernelW");
  param_.stride_h = opdesc.GetAttr<int>("StrideH");
  param_.stride_w = opdesc.GetAttr<int>("StrideW");

  if (opdesc.HasAttr("fuse_relu")) {
    param_.fuse_relu = opdesc.GetAttr<bool>("fuse_relu");
  }
#ifdef LITE_WITH_XPU
  if (opdesc.HasAttr("__xpu__float_to_fix")) {
    param_.__xpu__float_to_fix = opdesc.GetAttr<bool>("__xpu__float_to_fix");
  }
  if (opdesc.HasAttr("__xpu__w_max")) {
    param_.__xpu__w_max = opdesc.GetAttr<float>("__xpu__w_max");
  }
#endif

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(var_conv_2d, paddle::lite::operators::VarConv2dOp);
