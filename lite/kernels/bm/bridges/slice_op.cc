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
#include <bmcompiler_op_code.h>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int SliceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);

  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  // input
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_dims = input->dims();
  const int64_t* input_shape_data =
      const_cast<const int64_t*>(&input_dims.data()[0]);
  std::vector<int32_t> i_input_shape_data(input_dims.size());
  for (size_t i = 0; i < input_dims.size(); i++) {
    i_input_shape_data[i] = static_cast<int>(input_shape_data[i]);
  }
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto axes = op_info->GetAttr<std::vector<int32_t>>("axes");
  auto starts = op_info->GetAttr<std::vector<int32_t>>("starts");
  auto ends = op_info->GetAttr<std::vector<int32_t>>("ends");

  std::vector<int32_t> begin_index(input_dims.size(), 0);
  std::vector<int32_t> end_index(input_dims.size(), -1);
  std::vector<int32_t> strides(input_dims.size(), 1);
  int32_t begin_mask = 0;
  int32_t end_mask = 0;
  for (size_t i = 0; i < input_dims.size(); i++) {
    begin_mask |= (1 << i);
    end_mask |= (1 << i);
  }
  for (size_t i = 0; i < axes.size(); i++) {
    begin_index[axes[i]] = starts[i];
    end_index[axes[i]] = ends[i];
    begin_mask &= ~(1 << axes[i]);
    end_mask &= ~(1 << axes[i]);
  }

  add_stride_slice_layer_v2(graph->GetCompilerHandle(),
                            static_cast<const char*>(input_var_name.c_str()),
                            const_cast<const int*>(&i_input_shape_data[0]),
                            input_dims.size(),
                            static_cast<const char*>(output_var_name.c_str()),
                            begin_index.data(),
                            end_index.data(),
                            strides.data(),
                            input_dims.size(),
                            begin_mask,
                            end_mask,
                            0,
                            0,
                            0);
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(slice,
                         kBM,
                         paddle::lite::subgraph::bm::SliceConverter);
