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
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  // output
  int32_t* shape[1];
  int32_t dim[1];
  const char* name[1];
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  shape[0] = &i_output_shape_data[0];
  name[0] = static_cast<const char*>(output_var_name.c_str());
  dim[0] = output_dims.size();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  CHECK(pooling_type == "max" || pooling_type == "avg");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ceil_mode = op_info->GetAttr<bool>("ceil_mode");
  bool average_exclusive = false;
  if (pooling_type == "avg") {
    average_exclusive = op_info->GetAttr<bool>("exclusive");
  }
  if (global_pooling) {
    paddings[0] = 0;
    paddings[1] = 0;
    ksize[0] = i_x_shape_data[2];
    ksize[1] = i_x_shape_data[3];
  }
  add_pooling_layer(
      graph->GetCompilerHandle(),
      const_cast<const int*>(&i_x_shape_data[0]),
      x_dims.size(),
      static_cast<const char*>(x_var_name.c_str()),
      1,
      shape,
      dim,
      name,
      ksize[0],
      ksize[1],
      paddings[0],
      paddings[0],
      paddings[1],
      paddings[1],
      strides[0],
      strides[1],
      (ksize[0] > 1 && ksize[1] > 1) && pooling_type == "max" ? 0 : 1,
      static_cast<int>(average_exclusive),
      static_cast<int>(global_pooling),
      static_cast<int>(ceil_mode),
      static_cast<const char*>(unique_op_name.c_str()),
      nullptr);
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kBM,
                         paddle::lite::subgraph::bm::PoolConverter);
