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

#include "lite/operators/gru_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GRUOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input)
  CHECK_OR_FALSE(param_.weight)
  CHECK_OR_FALSE(param_.batch_gate)
  CHECK_OR_FALSE(param_.batch_reset_hidden_prev)
  CHECK_OR_FALSE(param_.batch_hidden)
  CHECK_OR_FALSE(param_.hidden)

  auto input_dims = param_.input->dims();
  auto weight_dims = param_.weight->dims();
  int input_size = input_dims[1];
  int frame_size = weight_dims[0];
  CHECK_EQ_OR_FALSE(input_size, frame_size * 3)
  CHECK_EQ_OR_FALSE(weight_dims[1], frame_size * 3)

  if (param_.h0) {
    auto h0_dims = param_.h0->dims();
    CHECK_EQ_OR_FALSE(h0_dims[1], frame_size)
  }

  if (param_.bias) {
    auto bias_dims = param_.bias->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1)
    CHECK_EQ_OR_FALSE(bias_width, frame_size * 3)
  }

  return true;
}

bool GRUOpLite::InferShape() const {
  auto input_dims = param_.input->dims();
  auto weight_dims = param_.weight->dims();
  int frame_size = weight_dims[0];
  auto batch_size = input_dims[0];

  param_.batch_gate->Resize(input_dims);
  param_.batch_reset_hidden_prev->Resize(lite::DDim({batch_size, frame_size}));
  param_.batch_hidden->Resize(lite::DDim({batch_size, frame_size}));
  param_.hidden->Resize(lite::DDim({batch_size, frame_size}));

  *(param_.hidden->mutable_lod()) = param_.input->lod();
  return true;
}

bool GRUOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto input = op_desc.Input("Input").front();
  auto weight = op_desc.Input("Weight").front();
  auto batch_gate = op_desc.Output("BatchGate").front();
  auto batch_reset_hidden_prev = op_desc.Output("BatchResetHiddenPrev").front();
  auto batch_hidden = op_desc.Output("BatchHidden").front();
  auto hidden = op_desc.Output("Hidden").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  if (op_desc.Input("H0").size()) {
    auto h0 = op_desc.Input("H0").front();
    param_.h0 = scope->FindVar(h0)->GetMutable<lite::Tensor>();
  }
  param_.weight = scope->FindVar(weight)->GetMutable<lite::Tensor>();

  param_.batch_gate = scope->FindVar(batch_gate)->GetMutable<lite::Tensor>();
  param_.batch_reset_hidden_prev =
      scope->FindVar(batch_reset_hidden_prev)->GetMutable<lite::Tensor>();
  param_.batch_hidden =
      scope->FindVar(batch_hidden)->GetMutable<lite::Tensor>();
  param_.hidden = scope->FindVar(hidden)->GetMutable<lite::Tensor>();

  if (op_desc.HasInput("Bias")) {
    auto bias = op_desc.Input("Bias").front();
    param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  }

  param_.gate_activation = op_desc.GetAttr<std::string>("gate_activation");
  param_.activation = op_desc.GetAttr<std::string>("activation");
  param_.is_reverse = op_desc.GetAttr<bool>("is_reverse");
  param_.origin_mode = op_desc.GetAttr<bool>("origin_mode");

  return true;
}

#ifdef LITE_WITH_TRAIN
bool GRUGradOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.weight);
  CHECK_OR_FALSE(param_.batch_gate);
  CHECK_OR_FALSE(param_.batch_reset_hidden_prev);
  CHECK_OR_FALSE(param_.batch_hidden);
  CHECK_OR_FALSE(param_.hidden);
  CHECK_OR_FALSE(param_.hidden_grad);

  auto input_dims = param_.input->dims();
  auto weight_dims = param_.weight->dims();
  int input_size = input_dims[1];
  int frame_size = weight_dims[0];
  int weight_height = weight_dims[0];
  int weight_width = weight_dims[1];
  CHECK_EQ_OR_FALSE(input_size, frame_size * 3)
  CHECK_EQ_OR_FALSE(weight_height, frame_size)
  CHECK_EQ_OR_FALSE(weight_width, frame_size * 3)

  if (param_.h0) {
    auto h0_dims = param_.h0->dims();
    CHECK_EQ_OR_FALSE(h0_dims[1], frame_size)
  }

  if (param_.bias) {
    auto bias_dims = param_.bias->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1)
    CHECK_EQ_OR_FALSE(bias_width, frame_size * 3)
  }

  return true;
}

bool GRUGradOpLite::InferShape() const {
  if (param_.input_grad) param_.input_grad->Resize(param_.input->dims());
  if (param_.weight_grad) param_.weight_grad->Resize(param_.weight->dims());
  if (param_.h0_grad) param_.h0_grad->Resize(param_.h0->dims());
  if (param_.bias_grad) param_.bias_grad->Resize(param_.bias->dims());
  return true;
}

bool GRUGradOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto input = op_desc.Input("Input").front();
  auto weight = op_desc.Input("Weight").front();
  auto batch_gate = op_desc.Iutput("BatchGate").front();
  auto batch_reset_hidden_prev = op_desc.Iutput("BatchResetHiddenPrev").front();
  auto batch_hidden = op_desc.Iutput("BatchHidden").front();
  auto hidden = op_desc.Iutput("Hidden").front();
  auto hidden_grad = op_desc.Input(framework::GradVarName("Hidden")).front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.weight = scope->FindVar(weight)->GetMutable<lite::Tensor>();

  param_.batch_gate = scope->FindVar(batch_gate)->GetMutable<lite::Tensor>();
  param_.batch_reset_hidden_prev =
      scope->FindVar(batch_reset_hidden_prev)->GetMutable<lite::Tensor>();
  param_.batch_hidden =
      scope->FindVar(batch_hidden)->GetMutable<lite::Tensor>();
  param_.hidden = scope->FindVar(hidden)->GetMutable<lite::Tensor>();
  param_.hidden_grad = scope->FindVar(hidden_grad)->GetMutable<lite::Tensor>();

  if (op_desc.Input("H0").size()) {
    auto h0 = op_desc.Input("H0").front();
    param_.h0 = scope->FindVar(h0)->GetMutable<lite::Tensor>();
  }
  if (op_desc.HasInput("Bias")) {
    auto bias = op_desc.Input("Bias").front();
    param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  }

  if (op_desc.Output(framework::GradVarName("Input")).size()) {
    auto input_grad = op_desc.Output(framework::GradVarName("Input")).front();
    param_.input_grad = scope->FindVar(input_grad)->GetMutable<lite::Tensor>();
  }
  if (op_desc.Output(framework::GradVarName("Weight")).size()) {
    auto weight_grad = op_desc.Output(framework::GradVarName("Weight")).front();
    param_.weight_grad = scope->FindVar(wight_grad)->GetMutable<lite::Tensor>();
  }
  if (op_desc.Output(framework::GradVarName("Bias")).size()) {
    auto bias_grad = op_desc.Output(framework::GradVarName("Bias")).front();
    param_.bias_grad = scope->FindVar(bias_grad)->GetMutable<lite::Tensor>();
  }
  if (op_desc.Output(framework::GradVarName("H0")).size()) {
    auto h0_grad = op_desc.Output(framework::GradVarName("H0")).front();
    param_.h0_grad = scope->FindVar(h0_grad)->GetMutable<lite::Tensor>();
  }

  return true;
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gru, paddle::lite::operators::GRUOpLite)
#ifdef LITE_WITH_TRAIN
REGISTER_LITE_OP(gru_grad, paddle::lite::operators::GRUGradOpLite)
#endif
