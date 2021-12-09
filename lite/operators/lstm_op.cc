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

#include "lite/operators/lstm_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {
inline lite_api::ActivationType GetActivationType(const std::string &type) {
  if (type == "sigmoid") {
    return lite_api::ActivationType::kSigmoid;
  } else if (type == "sigmoid_v2") {
    return lite_api::ActivationType::kSigmoid_v2;
  } else if (type == "relu") {
    return lite_api::ActivationType::kRelu;
  } else if (type == "tanh") {
    return lite_api::ActivationType::kTanh;
  } else if (type == "tanh_v2") {
    return lite_api::ActivationType::kTanh_v2;
  } else if (type == "identity" || type == "") {
    return lite_api::ActivationType::kIndentity;
  }
  LOG(FATAL) << "The input type is not supported: " << type;
  return lite_api::ActivationType::kIndentity;
}
bool LstmOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.Weight);
  CHECK_OR_FALSE(param_.Bias);
  return true;
}

bool LstmOp::InferShapeImpl() const {
  auto in_dims = param_.Input->dims();
  if (param_.H0) {
    CHECK(param_.C0) << "lstm must has H0 and C0 in the same time";
    auto h_dims = param_.H0->dims();
    auto c_dims = param_.C0->dims();
    CHECK_EQ(h_dims, c_dims) << "H0 and C0 dims must be same";
  }
  int frame_size = in_dims[1] / 4;
  auto w_dims = param_.Weight->dims();
  CHECK_EQ(w_dims.size(), 2) << "weight dims should be 2";
  CHECK_EQ(w_dims[0], frame_size) << "weight first dims should be "
                                  << frame_size;
  CHECK_EQ(w_dims[1], 4 * frame_size) << "weight dims should be 4 * "
                                      << frame_size;
  auto b_dims = param_.Bias->dims();
  CHECK_EQ(b_dims.size(), 2) << "Bias dims should be 2";
  CHECK_EQ(b_dims[0], 1) << "Bias first dims should be 1";
  if (param_.use_peepholes) {
    CHECK_EQ(b_dims[1], 7 * frame_size) << "Bias second dim must be 7 * "
                                        << frame_size;
  } else {
    CHECK_EQ(b_dims[1], 4 * frame_size) << "Bias second dim must be 4 * "
                                        << frame_size;
  }
  DDimLite out_dims(std::vector<int64_t>{in_dims[0], frame_size});
  param_.Hidden->Resize(out_dims);
  param_.Cell->Resize(out_dims);
  param_.BatchCellPreAct->Resize(out_dims);
  param_.BatchGate->Resize(in_dims);

  auto hidden_lod = param_.Hidden->mutable_lod();
  *hidden_lod = param_.Input->lod();
  auto cell_lod = param_.Cell->mutable_lod();
  *cell_lod = param_.Input->lod();
  return true;
}

bool LstmOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.Input =
      scope->FindVar(opdesc.Input("Input").front())->GetMutable<lite::Tensor>();
  param_.Weight = scope->FindVar(opdesc.Input("Weight").front())
                      ->GetMutable<lite::Tensor>();
  param_.Bias =
      scope->FindVar(opdesc.Input("Bias").front())->GetMutable<lite::Tensor>();
  param_.Hidden = scope->FindVar(opdesc.Output("Hidden").front())
                      ->GetMutable<lite::Tensor>();
  param_.Cell =
      scope->FindVar(opdesc.Output("Cell").front())->GetMutable<lite::Tensor>();
  param_.BatchGate = scope->FindVar(opdesc.Output("BatchGate").front())
                         ->GetMutable<lite::Tensor>();
  param_.BatchCellPreAct =
      scope->FindVar(opdesc.Output("BatchCellPreAct").front())
          ->GetMutable<lite::Tensor>();
  CHECK(param_.Input);
  CHECK(param_.Weight);
  CHECK(param_.Bias);
  if (opdesc.Input("C0").size()) {
    param_.C0 =
        scope->FindVar(opdesc.Input("C0").front())->GetMutable<lite::Tensor>();
  }
  if (opdesc.Input("H0").size()) {
    param_.H0 =
        scope->FindVar(opdesc.Input("H0").front())->GetMutable<lite::Tensor>();
  }
  param_.use_peepholes = opdesc.GetAttr<bool>("use_peepholes");
  param_.is_reverse = opdesc.GetAttr<bool>("is_reverse");
  param_.gate_activation =
      GetActivationType(opdesc.GetAttr<std::string>("gate_activation"));
  param_.cell_activation =
      GetActivationType(opdesc.GetAttr<std::string>("cell_activation"));
  param_.candidate_activation =
      GetActivationType(opdesc.GetAttr<std::string>("candidate_activation"));

  // For int8
  const OpInfo *op_info = static_cast<const OpInfo *>(&opdesc);
  if (op_info != nullptr && op_info->HasAttr("enable_int8") &&
      op_info->GetAttr<bool>("enable_int8")) {
    param_.enable_int8 = true;
    param_.bit_length = opdesc.GetAttr<int>("bit_length");
    std::string weight_scale_name = "Weight0_scale";
    if (op_info->HasInputScale(weight_scale_name, true)) {
      param_.weight_scale = op_info->GetInputScale(weight_scale_name, true);
    }
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lstm, paddle::lite::operators::LstmOp);
