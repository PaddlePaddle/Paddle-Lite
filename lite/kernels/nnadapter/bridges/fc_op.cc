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

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_GE(input_dims.size(), 2UL);
  auto w_name = op_info->Input("W").front();
  auto w_scale_name = "W0_scale";
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int M = input_dims.Slice(0, in_num_col_dims).production();
  int K = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int N = w_dims[1];
  CHECK_EQ(K * N, w_dims.production());
  CHECK_EQ(out_dims.size(), 2);
  CHECK_EQ(out_dims[0], M);
  CHECK_EQ(out_dims[1], N);
  VLOG(5) << "input dims: " << input_dims << " w dims: " << w_dims
          << " out_dims: " << out_dims << " M: " << M << " K: " << K
          << " N: " << N;
  std::string activation_type;
  if (op_info->HasAttr("activation_type")) {
    activation_type = op_info->GetAttr<std::string>("activation_type");
  }
  CHECK(activation_type.empty()) << "Unsupport activation_type "
                                 << activation_type << " is found.";

  // Input operand
  CHECK(op_info->HasInputScale(input_scale_name, true));
  auto input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(input_name)) {
    input_operand = converter->GetOperand(input_name);
  } else {
    NNAdapterOperandType input_type;
    memset(&input_type, 0, sizeof(NNAdapterOperandType));
    input_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    input_type.symm_per_layer_params.scale = input_scale;
    ConvertDimensions(
        input_dims, input_type.dimensions, &input_type.dimension_count);
    input_operand = converter->AddOperand(&input_type, input_name);
  }

  // Weight operand
  CHECK(op_info->HasInputScale(w_scale_name, true));
  auto w_scale = op_info->GetInputScale(w_scale_name, true);
  bool is_per_channel = IsPerChannelScales(w_scale);
  VLOG(5) << "is_per_channel: " << is_per_channel;
  NNAdapterOperandType weight_type;
  memset(&weight_type, 0, sizeof(NNAdapterOperandType));
  weight_type.dimension_count = w_dims.size();
  // Transpose to [k, n] to [n, k]
  weight_type.dimensions[0] = static_cast<int32_t>(w_dims[1]);
  weight_type.dimensions[1] = static_cast<int32_t>(w_dims[0]);
  if (is_per_channel) {
    // Per channel
    weight_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
    weight_type.symm_per_channel_params.scales = &w_scale[0];
    weight_type.symm_per_channel_params.scale_count = w_scale.size();
    weight_type.symm_per_channel_params.channel_dim = 0;
  } else {
    // Per layer
    weight_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    weight_type.symm_per_layer_params.scale = w_scale[0];
  }
  auto weight_operand = converter->AddOperand(&weight_type, w_name);
  auto w_data = w->mutable_data<int8_t>();
  std::vector<int8_t> transpose_weight_data(w_dims.production(), 0);
  Transpose(w_data, &transpose_weight_data[0], {1, 0}, w_dims.Vectorize());
  converter->SetOperandCopyFrom(weight_operand,
                                &transpose_weight_data[0],
                                sizeof(int8_t) * transpose_weight_data.size());

  // Bias
  NNAdapterOperandType bias_type;
  memset(&bias_type, 0, sizeof(NNAdapterOperandType));
  std::vector<float> bias_scale(w_scale.size());
  for (size_t i = 0; i < w_scale.size(); i++) {
    bias_scale[i] = input_scale * w_scale[i];
  }
  if (is_per_channel) {
    // Per channel
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
    bias_type.symm_per_channel_params.scales = &bias_scale[0];
    bias_type.symm_per_channel_params.scale_count = bias_scale.size();
    bias_type.symm_per_channel_params.channel_dim = 0;
  } else {
    // Per layer
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER;
    bias_type.symm_per_layer_params.scale = bias_scale[0];
  }
  bias_type.dimension_count = 1;
  bias_type.dimensions[0] = static_cast<int32_t>(N);
  std::vector<int32_t> quant_bias_data(N, 0);
  std::string bias_name = out_name + "_dummy_bias";
  if (HasInput(op_info, scope, "Bias")) {
    bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(bias_dims.production(), N);
    auto* bias_data = bias->mutable_data<float>();
    Quantize(bias_data, N, bias_scale, &quant_bias_data[0]);
  }
  auto bias_operand = converter->AddOperand(&bias_type, bias_name);
  converter->SetOperandCopyFrom(bias_operand,
                                &quant_bias_data[0],
                                sizeof(int32_t) * quant_bias_data.size());

  // Fuse code operand
  NNAdapterOperandType int32_type;
  memset(&int32_type, 0, sizeof(NNAdapterOperandType));
  int32_type.precision = NNADAPTER_INT32;
  int32_type.dimension_count = 0;

  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  auto fuse_code_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      fuse_code_operand, &fuse_code_value, sizeof(int32_t));

  // Output operand
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  NNAdapterOperandType output_type;
  memset(&output_type, 0, sizeof(NNAdapterOperandType));
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symm_per_layer_params.scale = out_scale;
  output_type.dimension_count = 2;
  output_type.dimensions[0] = static_cast<int32_t>(M);
  output_type.dimensions[1] = static_cast<int32_t>(N);
  auto output_operand = converter->AddOperand(&output_type, out_name);

  // Fully connected layer
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, weight_operand, bias_operand, fuse_code_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto fc_operation = converter->AddOperation(NNADAPTER_FULLY_CONNECTED);
  converter->SetOperation(fc_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::FCConverter);
