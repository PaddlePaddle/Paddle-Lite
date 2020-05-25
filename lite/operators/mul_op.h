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

#pragma once
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class MulOpLite : public OpLite {
 public:
  MulOpLite() {}

  explicit MulOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    AttachParam(&param_);

    CHECK(!op_desc.Input("X").empty());
    CHECK(!op_desc.Input("Y").empty());
    CHECK(!op_desc.Output("Out").empty());

    auto input = op_desc.Input("X").front();
    auto W = op_desc.Input("Y").front();
    auto out = op_desc.Output("Out").front();
    auto *var = scope->FindVar(input);
    CHECK(var);
    param_.x = &var->Get<Tensor>();
    var = scope->FindVar(W);
    CHECK(var) << "no var called " << W;
    param_.y = &var->Get<Tensor>();
    var = scope->FindVar(out);
    CHECK(var) << "no var called " << out;
    param_.output = var->GetMutable<Tensor>();
    param_.x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
    param_.y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");
    return true;
  }

  std::string DebugString() const override { return "mul"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.x->dims());
    ch->filter_shape = ch->DimToStr(param_.y->dims());
    ch->output_shape = ch->DimToStr(param_.output->dims());
    // ch->remark = "";
    auto x_dims = param_.x->dims();
    auto y_dims = param_.y->dims();
    auto x_mat_dims = x_dims.Flatten2D(param_.x_num_col_dims);
    auto y_mat_dims = y_dims.Flatten2D(param_.y_num_col_dims);
    ch->macs = 1.f * x_mat_dims[0] * x_mat_dims[1] * y_mat_dims[1];
  }
#endif

 private:
  mutable MulParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
