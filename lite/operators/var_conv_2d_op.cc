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

bool VarConv2dOp::CheckShape() const {
  auto x_dims = param_.X->dims();
  CHECK_EQ(x_dims.size(), 2) << "The rank of X(Input) can't be less than 2.";
  auto w_dims = param_.W->dims();
  CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor";
  CHECK_EQ(w_dims[0], param_.output_channel)
      << "W dim[0] should be equal to OutputChannel";
  CHECK_EQ(w_dims[1], param_.input_channel * param_.kernel_h * param_.kernel_w)
      << "W dim[1] should be equal to InputChannel * KernelH * KernelW";
  LoD x_lod = param_.X->lod();
  CHECK_EQ(x_lod.empty(), false) << "The Input(X) must hold lod info.";
  CHECK_GE(x_lod.size(), 1) << "The Input(X)'s lod info is corrupted.";
  CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod[0].back()))
      << "The Input(X)'s lod info mismatches the actual tensor shape.";
  LoD row_lod = param_.ROW->lod();
  CHECK_EQ(row_lod.empty(), false) << "The Input(ROW) must hold lod info.";
  LoD col_lod = param_.COLUMN->lod();
  CHECK_EQ(col_lod.empty(), false)
      << "The Input(COLUMN) must hold lod info." return true;
}

bool VarConv2dOp::InferShape() const { return true; }

bool VarConv2dOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.ROW = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("ROW").front())->Get<lite::Tensor>());
  param_.COLUMN = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("COLUMN").front())->Get<lite::Tensor>());
  param_.W = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("W").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.Col =
      scope->FindVar(opdesc.Output("Col").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X) << "X(Input) of VarConv2dOP should not be null.";
  CHECK(param_.ROW) << "Input(ROW) of VarConv2dOP should not be null.";
  CHECK(param_.COLUMN) << "Input(COLUMN) of VarConv2dOP should not be null.";
  CHECK(param_.W) << "W(Input) of VarConv2dOP should not be null.";
  CHECK(param_.Out) << "Out(Output) of VarConv2dOP should not be null.";
  CHECK(param_.Col) << "Col(Output) of VarConv2dOP should not be null.";
  param_.output_channel = opdesc.GetAttr<int>("OutputChannel");
  param_.input_channel = opdesc.GetAttr<int>("InputChannel");
  param_.kernel_h = opdesc.GetAttr<int>("KernelH");
  param_.kernel_w = opdesc.GetAttr<int>("KernelW");
  param_.stride_h = opdesc.GetAttr<int>("StrideH");
  param_.stride_w = opdesc.GetAttr<int>("StrideW");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(var_conv_2d, paddle::lite::operators::VarConv2dOp);
