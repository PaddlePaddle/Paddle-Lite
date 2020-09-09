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
#include <math.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  int channel_size = x_dims[1];
  auto scale_var_name = op_info->Input("Scale").front();
  auto scale = scope->FindVar(scale_var_name)->GetMutable<lite::Tensor>();
  auto bias_var_name = op_info->Input("Bias").front();
  auto bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
  auto mean_var_name = op_info->Input("Mean").front();
  auto mean = scope->FindVar(mean_var_name)->GetMutable<lite::Tensor>();
  auto variance_var_name = op_info->Input("Variance").front();
  auto variance = scope->FindVar(variance_var_name)->GetMutable<lite::Tensor>();
  // output
  auto output_var_name = op_info->Output("Y").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto unique_bn_out_name = lite::subgraph::bm::UniqueName("batch_norm_out");
  auto* scale_data = scale->mutable_data<float>();
  auto* bias_data = bias->mutable_data<float>();
  auto* mean_data = mean->mutable_data<float>();
  auto* variance_data = variance->mutable_data<float>();

  float* new_bias = static_cast<float*>(malloc(bias->memory_size()));
  float* new_scale = static_cast<float*>(malloc(scale->memory_size()));
  CHECK(new_bias != nullptr);
  CHECK(new_scale != nullptr);

  for (int c = 0; c < channel_size; c++) {
    float inv_scale = 1.f / (std::sqrt(variance_data[c] + epsilon));
    new_bias[c] = bias_data[c] - inv_scale * scale_data[c] * mean_data[c];
    new_scale[c] = inv_scale * scale_data[c];
  }

  const int input_num = 1;
  int** shape = new int*[input_num];
  int* dim = new int[input_num];
  const char** name = new const char*[input_num];
  name[0] = static_cast<const char*>(x_var_name.c_str());
  dim[0] = x_dims.size();
  shape[0] = &i_x_shape_data[0];
  add_scale_layer(graph->GetCompilerHandle(),
                  input_num,
                  shape,
                  dim,
                  name,
                  const_cast<const int*>(&i_output_shape_data[0]),
                  output_dims.size(),
                  static_cast<const char*>(output_var_name.c_str()),
                  static_cast<const char*>(unique_op_name.c_str()),
                  static_cast<const float*>(new_scale),
                  static_cast<const float*>(new_bias),
                  1,
                  1,
                  1);
  free(new_scale);
  free(new_bias);
  delete[] shape;
  delete[] name;
  delete[] dim;

  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kBM,
                         paddle::lite::subgraph::bm::BatchNormConverter);
