// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__spatial_transformer_resblock_op.h"
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

static std::vector<std::vector<int>> IntVec1DTo2D(const std::vector<int>& vec,
                                                  int dim) {
  std::vector<std::vector<int>> res;
  for (size_t i = 0; i < vec.size(); i += dim) {
    std::vector<int> tmp;
    for (size_t j = 0; j < dim; j++) {
      tmp.push_back(vec[i + j]);
    }
    res.emplace_back(std::move(tmp));
  }
  return res;
}

bool XPUSpatialTransformerResBlockOp::CheckShape() const { return true; }

bool XPUSpatialTransformerResBlockOp::InferShapeImpl() const {
  auto input_shape = param_.input1->dims();
  auto batch_size = input_shape[0];
  auto channel_out = param_.filter_dims[0][0];
  auto h = input_shape[2];
  auto w = input_shape[3];
  param_.output->Resize({batch_size, channel_out, h, w});
  return true;
}

bool XPUSpatialTransformerResBlockOp::AttachImpl(const cpp::OpDesc& op_desc,
                                                 lite::Scope* scope) {
  param_.has_silu_fc_input = op_desc.GetAttr<bool>("HasSiluFCInput");
  param_.include_silu = op_desc.GetAttr<bool>("IncludeSilu");
  param_.input1 = scope->FindTensor(op_desc.Input("Input1").front());
  if (param_.has_silu_fc_input) {
    param_.input2 = scope->FindTensor(op_desc.Input("Input2").front());
  }
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());

  param_.input_max.clear();
  for (auto& name : op_desc.Input("InputMax")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.input_max.push_back(t);
  }

  param_.fc_weight.clear();
  param_.fc_bias.clear();
  param_.weight_max.clear();
  if (param_.has_silu_fc_input) {
    for (auto& name : op_desc.Input("FCWeight")) {
      auto t = scope->FindVar(name)->GetMutable<Tensor>();
      param_.fc_weight.push_back(t);
    }
    for (auto& name : op_desc.Input("FCBias")) {
      auto t = scope->FindVar(name)->GetMutable<Tensor>();
      param_.fc_bias.push_back(t);
    }
    for (const auto& weight_max_tensor :
         op_desc.GetAttr<std::vector<std::string>>("FCWeightMax")) {
      auto tensor = scope->FindMutableTensor(weight_max_tensor);
      CHECK(tensor != nullptr);
      param_.weight_max.push_back(tensor);
    }
  }
  param_.conv_filter.clear();
  for (auto& name : op_desc.Input("ConvFilter")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.conv_filter.push_back(t);
  }
  param_.conv_bias.clear();
  for (auto& name : op_desc.Input("ConvBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.conv_bias.push_back(t);
  }
  param_.gn_scale.clear();
  for (auto& name : op_desc.Input("GNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_scale.push_back(t);
  }
  param_.gn_bias.clear();
  for (auto& name : op_desc.Input("GNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_bias.push_back(t);
  }
  param_.filter_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("ConvFilterMax")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.filter_max.push_back(tensor);
  }
  param_.groups = op_desc.GetAttr<std::vector<int>>("Groups");
  param_.strides =
      IntVec1DTo2D(op_desc.GetAttr<std::vector<int>>("Strides"), 2);
  param_.paddings =
      IntVec1DTo2D(op_desc.GetAttr<std::vector<int>>("Paddings"), 4);
  param_.dilations =
      IntVec1DTo2D(op_desc.GetAttr<std::vector<int>>("Dilations"), 2);
  param_.filter_dims =
      IntVec1DTo2D(op_desc.GetAttr<std::vector<int>>("FilterDims"), 4);
  param_.gn_groups = op_desc.GetAttr<std::vector<int>>("GNGroups");
  param_.gn_eps = op_desc.GetAttr<std::vector<float>>("GNEps");
  param_.conv_fix = op_desc.GetAttr<bool>("ConvFix");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__spatial_transformer_resblock,
                 paddle::lite::operators::XPUSpatialTransformerResBlockOp);
