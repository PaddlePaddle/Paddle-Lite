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
#include <user_bmcpu_common.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

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
  std::string pooling_type;
  if (op_info->HasAttr("pooling_type")) {
    pooling_type = op_info->GetAttr<std::string>("pooling_type");
  } else if (op_type == "max_pool2d_with_index") {
    pooling_type = "max";
  }
  CHECK(pooling_type == "max" || pooling_type == "avg");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  bool ceil_mode = false;
  if (op_info->HasAttr("ceil_mode")) {
    ceil_mode = op_info->GetAttr<bool>("ceil_mode");
  }
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  bool average_exclusive = false;
  if (pooling_type == "avg") {
    average_exclusive = op_info->GetAttr<bool>("exclusive");
  }
  if (output_dims[2] == 1 && output_dims[3] == 1) {
    global_pooling = true;
  }
  if (global_pooling) {
    paddings[0] = 0;
    paddings[1] = 0;
    ksize[0] = i_x_shape_data[2];
    ksize[1] = i_x_shape_data[3];
  }
  bool is_max = (pooling_type == "max");
  if (adaptive && !global_pooling) {
    user_cpu_param_t bm_param;
    bm_param.op_type = USER_PADDLE_ADAPTIVE_POOL;
    bm_param.u.adaptive_pool_parm.is_avg = !is_max;
    int32_t* in_shape[1];
    int32_t in_dim[1];
    const char* in_name[1];
    in_shape[0] = &i_x_shape_data[0];
    in_name[0] = static_cast<const char*>(x_var_name.c_str());
    in_dim[0] = x_dims.size();
    add_user_cpu_layer(graph->GetCompilerHandle(),
                       1,
                       in_shape,
                       in_dim,
                       in_name,
                       1,
                       shape,
                       dim,
                       name,
                       &bm_param,
                       static_cast<int>(sizeof(bm_param)));
  } else {
    add_pooling_layer(graph->GetCompilerHandle(),
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
                      is_max ? 0 : 1,
                      static_cast<int>(average_exclusive),
                      static_cast<int>(global_pooling),
                      static_cast<int>(ceil_mode),
                      static_cast<const char*>(unique_op_name.c_str()),
                      nullptr);
  }
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
REGISTER_SUBGRAPH_BRIDGE(max_pool2d_with_index,
                         kBM,
                         paddle::lite::subgraph::bm::PoolConverter);
