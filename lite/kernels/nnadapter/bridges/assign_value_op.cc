// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int AssignValueConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "Converting " << op_type << " ...";

  // Shape
  std::vector<int> shape_data = op_info->GetAttr<std::vector<int>>("shape");
  // Dtype
  auto dtype = op_info->GetAttr<int>("dtype");

  // Value Operand
  std::vector<float> fp32_values{};
  std::vector<int> int32_values{};
  std::vector<int64_t> int64_values{};
  std::vector<int> bool_values{};
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

  NNAdapterOperand* values_operand = nullptr;
  if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
    values_operand = converter->AddInt32ConstantOperand(
        &int32_values[0], DDim({static_cast<int>(shape_data.size())}));
  } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
    values_operand = converter->AddFloat32ConstantOperand(
        &fp32_values[0], DDim({static_cast<int>(shape_data.size())}));
  } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
    values_operand = converter->AddInt64ConstantOperand(
        &int64_values[0], DDim({static_cast<int>(shape_data.size())}));
  } else if (dtype == static_cast<int>(lite::core::FluidType::BOOL)) {
    values_operand = converter->AddInt32ConstantOperand(
        &bool_values[0], DDim({static_cast<int>(shape_data.size())}));
  } else {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Unsupported dtype for assign_value_op:"
                 << dtype;
    return FAILED;
  }
  CHECK(values_operand);

  // Output operand
  auto out_name = op_info->Output("Out").front();

  // Remapping values_operand to output operand
  converter->AddOperand(values_operand, out_name);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    assign_value,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::AssignValueConverter);
