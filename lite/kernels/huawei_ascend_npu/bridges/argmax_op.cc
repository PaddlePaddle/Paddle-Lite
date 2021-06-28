// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int ArgMaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  // 1. prepare input1: X node
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // 2. prepare input2: dimension node
  int32_t axis = op_info->GetAttr<int64_t>("axis");
  auto dimension_node = graph->Add<int32_t>(x_name + "/axis", axis);

  // 3. prepare output:
  auto out_name = op_info->Output("Out").front();
  auto argmax_node = graph->Add<ge::op::ArgMaxV2>(out_name);

  // 4. deal ascend unsupport attributes
  // 4.1 keepDims
  bool keepdims = false;
  if (op_info->HasAttr("keepdims")) {
    keepdims = op_info->GetAttr<bool>("keepdims");
  }
  if (keepdims) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Attr[keepdim]=true is not support";
    return FAILED;
  }

  // 5. prepare ascend attributes
  // 5.1 dtype: The output type, either "int32" or "int64". Defaults to "int64".
  int dtype = -1;
  if (op_info->HasAttr("dtype")) {
    dtype = op_info->GetAttr<int>("dtype");
  }

  auto ge_dtype = ge::DT_INT64;
  if (dtype == -1 || dtype == 3) {
    ge_dtype = ge::DT_INT64;
  } else if (dtype == 2) {
    ge_dtype = ge::DT_INT32;
  }

  // 6. pack op
  auto argmax_op = argmax_node->data<ge::op::ArgMaxV2>();
  argmax_op->set_input_x(*x_node->data());
  argmax_op->set_input_dimension(*dimension_node->data());
  argmax_op->set_attr_dtype(ge_dtype);
  INPUT_UPDATE(argmax_op, x, x_node);
  INPUT_UPDATE(argmax_op, dimension, dimension_node);
  OUTPUT_UPDATE(argmax_op, y, argmax_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    arg_max,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ArgMaxConverter);
