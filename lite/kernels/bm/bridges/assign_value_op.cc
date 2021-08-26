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

#include <bmcompiler_defs.h>
#include <bmcompiler_if.h>
#include <bmcompiler_if_lite.h>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int AssignValueConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();

  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  int buffer_size = 1;
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_dims[i]);
    buffer_size *= i_output_shape_data[i];
  }
  std::vector<float> fp32_values;
  std::vector<int> int32_values;
  float* assign_data =
      reinterpret_cast<float*>(malloc(buffer_size * sizeof(float)));
  CHECK(assign_data != nullptr);
  bm_data_type_t data_type = static_cast<bm_data_type_t>(DTYPE_FP32);
  fp32_values = op_info->GetAttr<std::vector<float>>("fp32_values");
  if (0 != fp32_values.size()) {
    for (int i = 0; i < fp32_values.size(); i++) {
      assign_data[i] = fp32_values[i];
    }
  } else {
    int32_values = op_info->GetAttr<std::vector<int>>("int32_values");
    data_type = static_cast<bm_data_type_t>(DTYPE_INT32);
    CHECK_EQ(buffer_size, int32_values.size());
    for (int i = 0; i < int32_values.size(); i++) {
      assign_data[i] = int32_values[i];
    }
  }

  bm_add_const_tensor(graph->GetCompilerHandle(),
                      static_cast<const char*>(output_var_name.c_str()),
                      const_cast<const int*>(i_output_shape_data.data()),
                      output_dims.size(),
                      data_type,
                      reinterpret_cast<const void*>(assign_data));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(assign_value,
                         kBM,
                         paddle::lite::subgraph::bm::AssignValueConverter);
