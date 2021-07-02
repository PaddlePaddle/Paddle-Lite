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
#include "lite/core/types.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int AssignValueConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  std::vector<float> fp32_values{};
  std::vector<int> int32_values{};
  std::vector<int64_t> int64_values{};
  std::vector<int> bool_values{};

  auto shape = op_info->GetAttr<std::vector<int>>("shape");
  std::vector<int64_t> shape_int64{};
  for (int i = 0; i < shape.size(); i++) {
    shape_int64.push_back(shape[i]);
  }

  auto dtype = op_info->GetAttr<int>("dtype");
  if (op_info->HasAttr("fp32_values")) {
    fp32_values = op_info->GetAttr<std::vector<float>>("fp32_values");
  }
  if (op_info->HasAttr("int32_values")) {
    int32_values = op_info->GetAttr<std::vector<int>>("int32_values");
  }
  if (op_info->HasAttr("int64_values")) {
    int64_values = op_info->GetAttr<std::vector<int64_t>>("int64_values");
  }
  if (op_info->HasAttr("bool_values")) {
    bool_values = op_info->GetAttr<std::vector<int>>("bool_values");
  }

  // Clip node
  auto out_name = op_info->Output("Out").front();
  if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
    graph->Add<int>(out_name, int32_values, shape_int64);
  } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
    graph->Add<float>(out_name, fp32_values, shape_int64);
  } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
    graph->Add<int64_t>(out_name, int64_values, shape_int64);
  } else if (dtype == static_cast<int>(lite::core::FluidType::BOOL)) {
    graph->Add<int>(out_name, bool_values, shape_int64);
  } else {
    LOG(FATAL) << "[HUAWEI_ASCEND_NPU] Unsupported dtype for assign_value_op:"
               << dtype;
    return FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    assign_value,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::AssignValueConverter);
