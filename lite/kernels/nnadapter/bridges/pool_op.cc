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
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  std::vector<int> ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
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
  bool ceil_mode =
      op_info->HasAttr("ceil_mode") && op_info->GetAttr<bool>("ceil_mode");
  // Exclusive
  bool exclusive =
      op_info->HasAttr("exclusive") && op_info->GetAttr<bool>("exclusive");

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    if (has_x_scale) {
      input_operand =
          converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input_operand = converter->AddFloat32VariableOperand(x_dims, x_name);
    }
  }

  // Auto_pad operand
  auto auto_pad_operand = converter->AddInt32ConstantOperand(
      static_cast<int32_t>(PaddingAlgorithm2PadCode(padding_algorithm)));

  // Pads operand(optional)
  auto pads_operand = converter->AddInt32ConstantOperand(
      paddings.data(), DDim({static_cast<int64_t>(paddings.size())}));

  // Kernel_shape operand
  if (global_pooling) {
    ksize[0] = static_cast<int32_t>(x_dims[2]);
    ksize[1] = static_cast<int32_t>(x_dims[3]);
  }
  auto kernel_shape_operand = converter->AddInt32ConstantOperand(
      ksize.data(), DDim({static_cast<int64_t>(ksize.size())}));

  // Strides operand
  auto strides_operand = converter->AddInt32ConstantOperand(
      strides.data(), DDim({static_cast<int64_t>(strides.size())}));

  // Ceil_mode(optional)
  auto ceil_mode_operand = converter->AddBool8ConstantOperand(ceil_mode);

  // Count_include_pad(optional), only for avg_pool
  auto count_include_pad_operand =
      converter->AddBool8ConstantOperand(!exclusive);

  // Return_indices(optional), only for max_pool
  auto return_indices_operand = converter->AddBool8ConstantOperand(false);

  // Fuse code operand
  auto fuse_code_operand =
      converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // 2-D Pooling operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand,
      auto_pad_operand,
      pads_operand,
      kernel_shape_operand,
      strides_operand,
      ceil_mode_operand,
      fuse_code_operand,
  };
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperationType pool2d_operation_type;
  if (pooling_type == "max") {
    pool2d_operation_type = NNADAPTER_MAX_POOL_2D;
    input_operands.insert(input_operands.begin() + 6, return_indices_operand);
    output_operands.push_back(nullptr);
  } else if (pooling_type == "avg") {
    pool2d_operation_type = NNADAPTER_AVERAGE_POOL_2D;
    input_operands.insert(input_operands.begin() + 6,
                          count_include_pad_operand);
  } else {
    LOG(WARNING) << "Unsupported pooling type: " << pooling_type;
    return FAILED;
  }
  converter->AddOperation(
      pool2d_operation_type, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::PoolConverter);
