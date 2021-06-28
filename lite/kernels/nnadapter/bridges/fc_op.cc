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
  auto has_input_scale = op_info->HasInputScale(input_scale_name, true);
  auto input_scale =
      has_input_scale ? op_info->GetInputScale(input_scale_name, true)[0] : 0.f;
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_GE(input_dims.size(), 2UL);
  auto w_name = op_info->Input("W").front();
  auto w_scale_name = "W0_scale";
  auto has_w_scale = op_info->HasInputScale(w_scale_name, true);
  auto w_scale = has_w_scale ? op_info->GetInputScale(w_scale_name, true)
                             : std::vector<float>({});
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int64_t M = input_dims.Slice(0, in_num_col_dims).production();
  int64_t K = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int64_t N = w_dims[1];
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
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(input_name)) {
    input_operand = converter->GetOperand(input_name);
  } else {
    if (has_input_scale) {
      input_operand = converter->AddQuant8VariableOperand(
          input_dims, input_scale, input_name);
    } else {
      input_operand =
          converter->AddFloat32VariableOperand(input_dims, input_name);
    }
  }

  // Weight operand
  // Transpose to [k, n] to [n, k]
  NNAdapterOperand* weight_operand = nullptr;
  bool is_per_channel = false;
  if (has_w_scale) {
    is_per_channel = IsPerChannelScales(w_scale);
    VLOG(5) << "is_per_channel: " << is_per_channel;
    auto w_data = w->mutable_data<int8_t>();
    std::vector<int8_t> transpose_weight_data(w_dims.production(), 0);
    DDim transpose_w_dims;
    Transpose(
        w_data, &transpose_weight_data[0], {1, 0}, w_dims, &transpose_w_dims);
    if (is_per_channel) {
      weight_operand =
          converter->AddQuant8ConstantOperand(&transpose_weight_data[0],
                                              transpose_w_dims,
                                              &w_scale[0],
                                              w_scale.size());
    } else {
      weight_operand = converter->AddQuant8ConstantOperand(
          &transpose_weight_data[0], transpose_w_dims, w_scale[0]);
    }
  } else {
    auto w_data = w->mutable_data<float>();
    std::vector<float> transpose_weight_data(w_dims.production(), 0);
    DDim transpose_w_dims;
    Transpose(
        w_data, &transpose_weight_data[0], {1, 0}, w_dims, &transpose_w_dims);
    weight_operand = converter->AddFloat32ConstantOperand(
        &transpose_weight_data[0], transpose_w_dims);
  }

  // Bias
  std::string bias_name = out_name + "_dummy_bias";
  float* bias_data = nullptr;
  if (HasInput(op_info, scope, "Bias")) {
    bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(bias_dims.production(), N);
    bias_data = bias->mutable_data<float>();
  }
  DDim bias_dims({N});
  NNAdapterOperand* bias_operand = nullptr;
  if (has_input_scale && has_w_scale) {
    std::vector<float> bias_scale(w_scale.size());
    for (size_t i = 0; i < w_scale.size(); i++) {
      bias_scale[i] = input_scale * w_scale[i];
    }
    std::vector<int32_t> quant_bias_data(N, 0);
    if (bias_data) {
      Quantize(bias_data, N, bias_scale, &quant_bias_data[0]);
    }
    if (is_per_channel) {
      bias_operand = converter->AddQuant32ConstantOperand(
          &quant_bias_data[0], bias_dims, &bias_scale[0], bias_scale.size());
    } else {
      bias_operand = converter->AddQuant32ConstantOperand(
          &quant_bias_data[0], bias_dims, bias_scale[0]);
    }
  } else {
    if (bias_data) {
      bias_operand =
          converter->AddFloat32ConstantOperand(bias_data, bias_dims, false);
    } else {
      // Dummy bias
      std::vector<float> dummy_bias_data(N, 0);
      bias_operand =
          converter->AddFloat32ConstantOperand(&dummy_bias_data[0], bias_dims);
    }
  }

  // Fuse code operand
  auto fuse_code_operand =
      converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(DDim({M, N}), out_scale, out_name);
  } else {
    output_operand =
        converter->AddFloat32VariableOperand(DDim({M, N}), out_name);
  }

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
