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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_dims = input->dims();
  auto output_var_name = op_info->Output("Output").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();

  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(output_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  bool has_bias = lite::subgraph::bm::HasInputArg(op_info, scope, "Bias");
  float* bias_data = nullptr;
  if (has_bias) {
    auto bias_var_name = op_info->Input("Bias").front();
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    bias_data = static_cast<float*>(bias->mutable_data<float>());
  }
  const int64_t* input_shape_data =
      const_cast<const int64_t*>(&input_dims.data()[0]);
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_input_shape_data(input_dims.size());
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < input_dims.size(); i++) {
    i_input_shape_data[i] = static_cast<int32_t>(input_shape_data[i]);
  }
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int32_t>(output_shape_data[i]);
  }
  const float* filter_data =
      const_cast<const float*>(filter->mutable_data<float>());
  auto groups = op_info->GetAttr<int>("groups");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  add_conv_layer(graph->GetCompilerHandle(),
                 const_cast<const int*>(&i_input_shape_data[0]),
                 input_dims.size(),
                 static_cast<const char*>(input_var_name.c_str()),
                 const_cast<const int*>(&i_output_shape_data[0]),
                 output_dims.size(),
                 static_cast<const char*>(output_var_name.c_str()),
                 static_cast<const char*>(unique_op_name.c_str()),
                 filter_data,
                 bias_data,
                 filter_dims.data()[2],
                 filter_dims.data()[3],
                 groups,
                 paddings[0],
                 paddings[0],
                 paddings[1],
                 paddings[1],
                 strides[0],
                 strides[1],
                 dilations[0],
                 dilations[1],
                 static_cast<int>(has_bias));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kBM,
                         paddle::lite::subgraph::bm::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kBM,
                         paddle::lite::subgraph::bm::ConvConverter);
