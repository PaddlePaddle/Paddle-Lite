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

int FillConstantConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  std::vector<int64_t> shape{};
  float fp32_value{0.f};

  auto out_name = op_info->Output("Out").front();
  std::shared_ptr<Node> out_shape_node{nullptr};
  if (op_info->HasAttr("shape")) {
    shape = op_info->GetAttr<std::vector<int64_t>>("shape");
    out_shape_node = graph->Add<int64_t>(out_name + "/dims", shape);
  }

  if (op_info->HasInput("ShapeTensor") &&
      !op_info->Input("ShapeTensor").empty()) {
    LOG(WARNING)
        << "[HUAWEI_ASCEND_NPU] Unsupported ShapeTensorList Input for " +
               op_type + "...";
    return FAILED;
  }

  if (op_info->HasInput("ShapeTensorList") &&
      !op_info->Input("ShapeTensorList").empty()) {
    LOG(WARNING)
        << "[HUAWEI_ASCEND_NPU] Unsupported ShapeTensorList Input for " +
               op_type + "...";
    return FAILED;
  }

  std::shared_ptr<Node> value_node{nullptr};
  fp32_value =
      op_info->HasAttr("value") ? op_info->GetAttr<float>("value") : 0.0f;
  if (op_info->HasInput("ValueTensor") &&
      !op_info->Input("ValueTensor").empty()) {
    auto value_tensor_name = op_info->Input("ValueTensor").front();
    auto value_tensor = scope->FindMutableTensor(value_tensor_name);
    value_node = graph->Add(value_tensor_name, *value_tensor);
  } else {
    value_node = graph->Add<float>(out_name + "/value", fp32_value);
  }

  auto fill_constant_node = graph->Add<ge::op::Fill>(out_name);
  auto fill_constant_op = fill_constant_node->data<ge::op::Fill>();
  fill_constant_op->set_input_dims(*out_shape_node->data());
  fill_constant_op->set_input_value(*value_node->data());
  INPUT_UPDATE(fill_constant_op, dims, out_shape_node);
  INPUT_UPDATE(fill_constant_op, value, value_node);
  OUTPUT_UPDATE(fill_constant_op, y, fill_constant_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fill_constant,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::FillConstantConverter);
