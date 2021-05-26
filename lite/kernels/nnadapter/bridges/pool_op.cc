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

#include "lite/operators/pool_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  // Check pooling mode
  if (pooling_type == "max" || pooling_type == "avg") {
  } else {
    LOG(WARNING) << "Unsupported pooling type: " << pooling_type;
    return FAILED;
  }
  // Calculate paddings and strides
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); i++) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "Paddings size should be the same or twice as the input size.";
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);
  // Ceil mode
  int8_t ceil_mode =
      op_info->HasAttr("ceil_mode") && op_info->GetAttr<bool>("ceil_mode");
  // Exclusive
  bool exclusive =
      op_info->HasAttr("exclusive") && op_info->GetAttr<bool>("exclusive");

  // Input operand
  CHECK(op_info->HasInputScale(x_scale_name, true));
  auto x_scale = op_info->GetInputScale(x_scale_name, true)[0];
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    NNAdapterOperandType input_type;
    memset(&input_type, 0, sizeof(NNAdapterOperandType));
    input_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    input_type.symm_per_layer_params.scale = x_scale;
    ConvertDimensions(
        x_dims, input_type.dimensions, &input_type.dimension_count);
    input_operand = converter->AddOperand(&input_type, x_name);
  }

  // Paddings and strides operands
  NNAdapterOperandType int32_type;
  memset(&int32_type, 0, sizeof(NNAdapterOperandType));
  int32_type.precision = NNADAPTER_INT32;
  int32_type.dimension_count = 0;

  auto padding_width_left_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_width_left_operand, &paddings[2], sizeof(int32_t));

  auto padding_width_right_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_width_right_operand, &paddings[3], sizeof(int32_t));

  auto padding_height_top_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_height_top_operand, &paddings[0], sizeof(int32_t));

  auto padding_height_bottom_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_height_bottom_operand, &paddings[1], sizeof(int32_t));

  auto stride_width_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      stride_width_operand, &strides[1], sizeof(int32_t));

  auto stride_height_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      stride_height_operand, &strides[0], sizeof(int32_t));

  auto filter_width_operand = converter->AddOperand(&int32_type);
  int32_t filter_width = global_pooling ? x_dims[3] : ksize[1];
  converter->SetOperandCopyFrom(
      filter_width_operand, &filter_width, sizeof(int32_t));

  auto filter_height_operand = converter->AddOperand(&int32_type);
  int32_t filter_height = global_pooling ? x_dims[2] : ksize[0];
  converter->SetOperandCopyFrom(
      filter_height_operand, &filter_height, sizeof(int32_t));

  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  auto fuse_code_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      fuse_code_operand, &fuse_code_value, sizeof(int32_t));

  NNAdapterOperandType bool8_type;
  memset(&bool8_type, 0, sizeof(NNAdapterOperandType));
  bool8_type.precision = NNADAPTER_BOOL8;
  bool8_type.dimension_count = 0;

  auto ceil_mode_operand = converter->AddOperand(&bool8_type);
  converter->SetOperandCopyFrom(ceil_mode_operand, &ceil_mode, sizeof(int8_t));

  int8_t count_include_pad = exclusive ? 0 : 1;
  auto count_include_pad_operand = converter->AddOperand(&bool8_type);
  converter->SetOperandCopyFrom(
      count_include_pad_operand, &count_include_pad, sizeof(int8_t));

  // Output operand
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  NNAdapterOperandType output_type;
  memset(&output_type, 0, sizeof(NNAdapterOperandType));
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symm_per_layer_params.scale = out_scale;
  ConvertDimensions(
      out_dims, output_type.dimensions, &output_type.dimension_count);
  auto output_operand = converter->AddOperand(&output_type, out_name);

  // 2-D Pooling operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand,
      padding_width_left_operand,
      padding_width_right_operand,
      padding_height_top_operand,
      padding_height_bottom_operand,
      stride_width_operand,
      stride_height_operand,
      filter_width_operand,
      filter_height_operand,
      fuse_code_operand,
      ceil_mode_operand,
      count_include_pad_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* pool2d_operation = nullptr;
  if (pooling_type == "max") {
    pool2d_operation = converter->AddOperation(NNADAPTER_MAX_POOL_2D);
  } else if (pooling_type == "avg") {
    pool2d_operation = converter->AddOperation(NNADAPTER_AVERAGE_POOL_2D);
  } else {
    LOG(WARNING) << "Unsupported pooling type: " << pooling_type;
    return FAILED;
  }
  converter->SetOperation(pool2d_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::PoolConverter);
