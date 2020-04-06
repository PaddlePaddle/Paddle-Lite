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

#include "lite/operators/gru_unit_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GRUUnitOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.hidden_prev);
  CHECK_OR_FALSE(param_.gate);
  CHECK_OR_FALSE(param_.reset_hidden_prev);
  CHECK_OR_FALSE(param_.hidden);
  CHECK_OR_FALSE(param_.weight);

  auto input_dims = param_.input->dims();
  auto hidden_prev_dims = param_.hidden_prev->dims();
  auto weight_dims = param_.weight->dims();

  int input_size = input_dims[1];
  int frame_size = hidden_prev_dims[1];
  int weight_height = weight_dims[0];
  int weight_width = weight_dims[1];
  CHECK_EQ_OR_FALSE(input_size, frame_size * 3)
  CHECK_EQ_OR_FALSE(weight_height, frame_size)
  CHECK_EQ_OR_FALSE(weight_width, frame_size * 3)

  if (param_.bias) {
    auto bias_dims = param_.bias->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1);
    CHECK_EQ_OR_FALSE(bias_width, frame_size * 3);
  }

  return true;
}

bool GRUUnitOpLite::InferShapeImpl() const {
  auto input_dims = param_.input->dims();
  auto hidden_prev_dims = param_.hidden_prev->dims();
  auto weight_dims = param_.weight->dims();

  int batch_size = input_dims[0];
  int frame_size = hidden_prev_dims[1];

  param_.gate->Resize(lite::DDim({batch_size, frame_size * 3}));
  param_.reset_hidden_prev->Resize(lite::DDim({batch_size, frame_size}));
  param_.hidden->Resize(lite::DDim({batch_size, frame_size}));

  auto out_lod = param_.hidden->mutable_lod();
  *out_lod = param_.input->lod();
  return true;
}

bool GRUUnitOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto input = op_desc.Input("Input").front();
  auto hidden_prev = op_desc.Input("HiddenPrev").front();
  auto weight = op_desc.Input("Weight").front();
  auto gate = op_desc.Output("Gate").front();
  auto reset_hidden_prev = op_desc.Output("ResetHiddenPrev").front();
  auto hidden = op_desc.Output("Hidden").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.hidden_prev = scope->FindVar(hidden_prev)->GetMutable<lite::Tensor>();
  param_.weight = scope->FindVar(weight)->GetMutable<lite::Tensor>();

  param_.gate = scope->FindVar(gate)->GetMutable<lite::Tensor>();
  param_.reset_hidden_prev =
      scope->FindVar(reset_hidden_prev)->GetMutable<lite::Tensor>();
  param_.hidden = scope->FindVar(hidden)->GetMutable<lite::Tensor>();

  if (op_desc.HasInput("Bias")) {
    auto bias = op_desc.Input("Bias").front();
    param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  }

  param_.gate_activation = op_desc.GetAttr<int>("gate_activation");
  param_.activation = op_desc.GetAttr<int>("activation");
  param_.origin_mode = op_desc.GetAttr<bool>("origin_mode");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gru_unit, paddle::lite::operators::GRUUnitOpLite)
