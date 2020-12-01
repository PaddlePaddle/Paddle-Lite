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
#include <bmcompiler_if.h>
#include <user_bmcpu_common.h>
#include <iostream>
#include <string>
#include <vector>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int BoxCoderConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);

  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto box_var_name = op_info->Input("PriorBox").front();
  auto box = scope->FindVar(box_var_name)->GetMutable<lite::Tensor>();
  auto box_dims = box->dims();
  auto box_var_var_name = op_info->Input("PriorBoxVar").front();
  auto box_var = scope->FindVar(box_var_var_name)->GetMutable<lite::Tensor>();
  auto box_var_dims = box_var->dims();
  auto target_box_var_name = op_info->Input("TargetBox").front();
  auto target_box =
      scope->FindVar(target_box_var_name)->GetMutable<lite::Tensor>();
  auto target_box_dims = target_box->dims();
  auto output_var_name = op_info->Output("OutputBox").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();

  std::vector<int32_t> i_box_shape_data(box_dims.size());
  for (size_t i = 0; i < box_dims.size(); i++) {
    i_box_shape_data[i] = static_cast<int32_t>(box_dims[i]);
  }
  std::vector<int32_t> i_box_var_shape_data(box_var_dims.size());
  for (size_t i = 0; i < box_var_dims.size(); i++) {
    i_box_var_shape_data[i] = static_cast<int32_t>(box_var_dims[i]);
  }
  std::vector<int32_t> i_target_box_shape_data(target_box_dims.size());
  for (size_t i = 0; i < target_box_dims.size(); i++) {
    i_target_box_shape_data[i] = static_cast<int32_t>(target_box_dims[i]);
  }
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int32_t>(output_dims[i]);
  }
  auto code_type = op_info->GetAttr<std::string>("code_type");
  auto box_normalized = op_info->GetAttr<bool>("box_normalized");
  int32_t axis = 0;
  if (op_info->HasAttr("axis")) {
    axis = op_info->GetAttr<int32_t>("axis");
  }
  std::vector<float> variance;
  if (op_info->HasAttr("variance")) {
    variance = op_info->GetAttr<std::vector<float>>("variance");
  }
  int variance_len = variance.size();
  user_cpu_param_t bm_param;
  bm_param.op_type = USER_PADDLE_BOX_CODER;
  bm_param.u.box_coder_param.axis = axis;
  CHECK_LE(variance_len, 20);
  memset(bm_param.u.box_coder_param.variance, 0, 20 * sizeof(float));
  memcpy(bm_param.u.box_coder_param.variance,
         &variance[0],
         variance_len * sizeof(float));
  bm_param.u.box_coder_param.variance_len = variance_len;
  bm_param.u.box_coder_param.code_type =
      (code_type == "encode_center_size") ? 0 : 1;
  bm_param.u.box_coder_param.normalized = box_normalized;
  int32_t input_num = 3;
  int32_t output_num = 1;
  int32_t* in_shape[3];
  int32_t in_dim[3];
  const char* in_name[3];
  in_shape[0] = &i_box_shape_data[0];
  in_shape[1] = &i_target_box_shape_data[0];
  in_shape[2] = &i_box_var_shape_data[0];
  in_dim[0] = box_dims.size();
  in_dim[1] = target_box_dims.size();
  in_dim[2] = box_var_dims.size();
  in_name[0] = static_cast<const char*>(box_var_name.c_str());
  in_name[1] = static_cast<const char*>(target_box_var_name.c_str());
  in_name[2] = static_cast<const char*>(box_var_var_name.c_str());
  int32_t* out_shape[1];
  int32_t out_dim[1];
  const char* out_name[1];
  out_shape[0] = &i_output_shape_data[0];
  out_dim[0] = output_dims.size();
  out_name[0] = static_cast<const char*>(output_var_name.c_str());

  add_user_cpu_layer(graph->GetCompilerHandle(),
                     input_num,
                     in_shape,
                     in_dim,
                     in_name,
                     output_num,
                     out_shape,
                     out_dim,
                     out_name,
                     &bm_param,
                     static_cast<int>(sizeof(bm_param)));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(box_coder,
                         kBM,
                         paddle::lite::subgraph::bm::BoxCoderConverter);
