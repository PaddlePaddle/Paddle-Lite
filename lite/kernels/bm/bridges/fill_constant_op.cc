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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int FillConstantConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  }
  float* const_data =
      reinterpret_cast<float*>(malloc(buffer_size * sizeof(float)));
  CHECK(const_data != nullptr);
  auto value = op_info->GetAttr<float>("value");
  for (size_t i = 0; i < buffer_size; i++) {
    const_data[i] = value;
  }
  bm_add_const_tensor(graph->GetCompilerHandle(),
                      static_cast<const char*>(output_var_name.c_str()),
                      const_cast<const int*>(i_output_shape_data.data()),
                      output_dims.size(),
                      static_cast<bm_data_type_t>(DTYPE_FP32),
                      reinterpret_cast<const void*>(const_data));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fill_constant,
                         kBM,
                         paddle::lite::subgraph::bm::FillConstantConverter);
