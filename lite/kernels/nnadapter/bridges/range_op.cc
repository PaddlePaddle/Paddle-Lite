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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

static void AddOperandByTensor(Converter* converter,
                               NNAdapterOperand** insert_operand,
                               Tensor* tensor,
                               const std::string& tensor_name) {
  auto tensor_precision = tensor->precision();
  switch (tensor_precision) {
    case PRECISION(kInt32): {
      *insert_operand = converter->AddInt32Operand(tensor, tensor_name);
      break;
    }
    case PRECISION(kInt64): {
      *insert_operand = converter->AddInt64Operand(tensor, tensor_name);
      break;
    }
    case PRECISION(kFP64): {
      *insert_operand = converter->AddFloat64Operand(tensor, tensor_name);
      break;
    }
    case PRECISION(kFloat):
    default: {
      *insert_operand = converter->AddFloat32Operand(tensor, tensor_name);
    }
  }
}

int RangeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto start_name = op_info->Input("Start").front();
  auto start = scope->FindMutableTensor(start_name);

  auto ends_name = op_info->Input("End").front();
  auto ends = scope->FindMutableTensor(ends_name);

  auto step_name = op_info->Input("Step").front();
  auto step = scope->FindMutableTensor(step_name);

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  // Start operand
  NNAdapterOperand* start_operand = nullptr;
  if (converter->HasOperand(start_name)) {
    start_operand = converter->GetOperand(start_name);
  } else {
    AddOperandByTensor(converter, &start_operand, start, start_name);
  }
  // Ends operand
  NNAdapterOperand* ends_operand = nullptr;
  if (converter->HasOperand(ends_name)) {
    ends_operand = converter->GetOperand(ends_name);
  } else {
    AddOperandByTensor(converter, &ends_operand, ends, ends_name);
  }
  // Step operand
  NNAdapterOperand* step_operand = nullptr;
  if (converter->HasOperand(step_name)) {
    step_operand = converter->GetOperand(step_name);
  } else {
    AddOperandByTensor(converter, &step_operand, step, step_name);
  }

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  AddOperandByTensor(converter, &output_operand, out, out_name);

  // Range operation
  std::vector<NNAdapterOperand*> input_operands = {
      start_operand, ends_operand, step_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* range_operation =
      converter->AddOperation(NNADAPTER_RANGE);
  converter->SetOperation(range_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(range,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::RangeConverter);
