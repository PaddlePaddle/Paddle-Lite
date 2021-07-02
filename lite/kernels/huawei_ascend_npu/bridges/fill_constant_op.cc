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

  VarDescAPI::VarDataType dtype = op_info->HasAttr("dtype")
                                      ? (static_cast<VarDescAPI::VarDataType>(
                                            op_info->GetAttr<int>("dtype")))
                                      : (VarDescAPI::VarDataType::FP32);

  ge::DataType o_dtype = ge::DT_FLOAT;
  PrecisionType p_type = PRECISION(kFloat);
  CvtType(dtype, &o_dtype, &p_type);
  std::vector<int64_t> shape{};

  float value =
      op_info->HasAttr("value") ? op_info->GetAttr<float>("value") : 0.0f;

  if (op_info->HasInput("ValueTensor") &&
      !op_info->Input("ValueTensor").empty()) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Unsupported ValueTensor Input for " +
                        op_type + "...";
    return FAILED;
  }

  if (op_info->HasInput("ShapeTensor") &&
      !op_info->Input("ShapeTensor").empty()) {
    auto shape_tensor = scope->FindTensor("ShapeTensor");
    auto shape_tensor_data = shape_tensor->data<int>();
    for (int i = 0; i < shape_tensor->numel(); i++) {
      shape.push_back(shape_tensor_data[i]);
    }
  } else if (op_info->HasAttr("shape")) {
    shape = op_info->GetAttr<std::vector<int64_t>>("shape");
  }

  if (op_info->HasInput("ShapeTensorList") &&
      !op_info->Input("ShapeTensorList").empty()) {
    LOG(WARNING)
        << "[HUAWEI_ASCEND_NPU] Unsupported ShapeTensorList Input for " +
               op_type + "...";
    return FAILED;
  }

  auto out_name = op_info->Output("Out").front();
  auto out_shape_node = graph->Add<int64_t>(out_name + "/dims", shape);

  auto fill_constant_node = graph->Add<ge::op::FillV2>(out_name, p_type);
  auto fill_constant_op = fill_constant_node->data<ge::op::FillV2>();
  fill_constant_op->set_input_dims(*out_shape_node->data());
  fill_constant_op->set_attr_value(value);
  INPUT_UPDATE(fill_constant_op, dims, out_shape_node);
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
