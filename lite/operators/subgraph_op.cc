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

bool SubgraphOp::InferShapeImpl() const {
  // Set output tensor lod
  // 1. All input lods should be the same
  // 2. All outputs share the input lod
  auto scope = param_.exec_scope;
  for (auto input_data_name : param_.input_data_names) {
    auto lod = scope->FindTensor(input_data_name)->lod();
    if (!lod.empty()) {
      for (auto output_data_name : param_.output_data_names) {
        scope->FindMutableTensor(output_data_name)->set_lod(lod);
      }
      break;
    }
  }
  return CheckShape();
}

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
  // Get the quantization parameters of input and output data variables
  auto op_info = static_cast<const OpInfo*>(&op_desc);
  param_.input_data_scales.clear();
  param_.output_data_scales.clear();
  for (auto& input_data_name : param_.input_data_names) {
    auto it = std::find(
        param_.input_names.begin(), param_.input_names.end(), input_data_name);
    CHECK(it != param_.input_names.end());
    int arg_index = it - param_.input_names.begin();
    std::string scale_name = "Inputs" + to_string(arg_index) + "_scale";
    float scale_value = -1.0f;
    if (op_info->HasInputScale(scale_name, true))
      scale_value = op_info->GetInputScale(scale_name, true)[0];
    param_.input_data_scales.emplace_back(scale_value);
  }
  for (auto& output_data_name : param_.output_data_names) {
    auto it = std::find(param_.output_names.begin(),
                        param_.output_names.end(),
                        output_data_name);
    CHECK(it != param_.output_names.end());
    int arg_index = it - param_.output_names.begin();
    std::string scale_name = "Outputs" + to_string(arg_index) + "_scale";
    float scale_value = -1.0f;
    if (op_info->HasOutputScale(scale_name, true))
      scale_value = op_info->GetOutputScale(scale_name, true)[0];
    param_.output_data_scales.emplace_back(scale_value);
  }
  CHECK(param_.program_desc);
  param_.block_idx = op_desc.GetAttr<int32_t>("sub_block");
  CHECK_GE(param_.block_idx, 0);
  param_.exec_scope = scope;
  CHECK(param_.exec_scope);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(subgraph, paddle::lite::operators::SubgraphOp);
