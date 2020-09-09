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

  const auto& input_dims = param_.input->dims();
  const auto& weight_dims = param_.weight->dims();
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

bool GRUOpLite::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  const auto& weight_dims = param_.weight->dims();
  int frame_size = weight_dims[0];
  auto batch_size = input_dims[0];

  param_.batch_gate->Resize(input_dims);

  DDim out_dims({batch_size, frame_size});
  param_.batch_reset_hidden_prev->Resize(out_dims);
  param_.batch_hidden->Resize(out_dims);
  param_.hidden->Resize(out_dims);

  *(param_.hidden->mutable_lod()) = param_.input->lod();
  return true;
}

bool GRUOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto input = op_desc.Input("Input").front();
  auto weight = op_desc.Input("Weight").front();
  auto batch_gate = op_desc.Output("BatchGate").front();
  auto batch_reset_hidden_prev = op_desc.Output("BatchResetHiddenPrev").front();
  auto batch_hidden = op_desc.Output("BatchHidden").front();
  auto hidden = op_desc.Output("Hidden").front();
  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  if (!op_desc.Input("H0").empty()) {
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

  if (!op_desc.Input("Bias").empty()) {
    auto bias = op_desc.Input("Bias").front();
    param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  }

  param_.gate_activation = op_desc.GetAttr<std::string>("gate_activation");
  param_.activation = op_desc.GetAttr<std::string>("activation");
  param_.is_reverse = op_desc.GetAttr<bool>("is_reverse");
  if (op_desc.HasAttr("origin_mode")) {
    param_.origin_mode = op_desc.GetAttr<bool>("origin_mode");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gru, paddle::lite::operators::GRUOpLite)
