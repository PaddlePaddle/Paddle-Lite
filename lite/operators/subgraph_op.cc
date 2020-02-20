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

#include "lite/operators/subgraph_op.h"
#include <utility>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SubgraphOp::CheckShape() const { return true; }

bool SubgraphOp::InferShape() const { return CheckShape(); /* enrich me */ }

bool SubgraphOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input_names = op_desc.Input("Inputs");
  param_.output_names = op_desc.Output("Outputs");
  for (auto& input_name : param_.input_names) {
    CHECK(scope->FindVar(input_name));
    scope->FindVar(input_name)->GetMutable<lite::Tensor>();
  }
  for (auto& output_name : param_.output_names) {
    CHECK(scope->FindVar(output_name));
    scope->FindVar(output_name)->GetMutable<lite::Tensor>();
  }
  param_.input_data_names =
      op_desc.GetAttr<std::vector<std::string>>("input_data_names");
  param_.output_data_names =
      op_desc.GetAttr<std::vector<std::string>>("output_data_names");
  CHECK(param_.sub_block_desc);
  param_.sub_block_idx = op_desc.GetAttr<int32_t>("sub_block");
  param_.scope = scope;
  CHECK(param_.scope);
  return true;
}

bool ResNet50Op::CheckShape() const { return true; }

bool ResNet50Op::InferShape() const {
  auto input_shape = param_.input->dims();
  input_shape[1] = 2048;
  input_shape[2] = 1;
  input_shape[3] = 1;
  param_.output->Resize(input_shape);
  return true;
}

bool ResNet50Op::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<lite::Tensor>();

  param_.filter.clear();
  for (auto& name : op_desc.Input("Filter")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.filter.push_back(t);
  }
  param_.bias.clear();
  for (auto& name : op_desc.Input("Bias")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.bias.push_back(t);
  }
  param_.max_filter.clear();
  for (auto& name : op_desc.Input("MaxFilter")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.max_filter.push_back(t);
  }
  return true;
}

bool MultiEncoderOp::CheckShape() const { return true; }

bool MultiEncoderOp::InferShape() const {
  auto input_shape = param_.input->dims();
  param_.output->Resize(input_shape);
  return true;
}

bool MultiEncoderOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.mask = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("Mask").front())->Get<lite::Tensor>());
  param_.fc_weight_max = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("FCWeightMax").front())->Get<lite::Tensor>());
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<lite::Tensor>();

  param_.fc_weight.clear();
  for (auto& name : op_desc.Input("FCWeight")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_weight.push_back(t);
  }
  param_.fc_bias.clear();
  for (auto& name : op_desc.Input("FCBias")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_bias.push_back(t);
  }
  param_.ln_scale.clear();
  for (auto& name : op_desc.Input("LNScale")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_scale.push_back(t);
  }
  param_.ln_bias.clear();
  for (auto& name : op_desc.Input("LNBias")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_bias.push_back(t);
  }

  param_.n_layers = op_desc.GetAttr<int>("n_layers");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.act_type = op_desc.GetAttr<std::string>("act_type");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(subgraph, paddle::lite::operators::SubgraphOp);
REGISTER_LITE_OP(ResNet50, paddle::lite::operators::ResNet50Op);
REGISTER_LITE_OP(MultiEncoder, paddle::lite::operators::MultiEncoderOp);
