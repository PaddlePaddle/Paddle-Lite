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

#include "lite/kernels/nnadapter/converter/converter.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertFC(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
  auto op_type = op->Type();
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto w_name = op->Input("W").front();
  auto w_scale_name = "W0_scale";
  auto w_tensor = scope->FindMutableTensor(w_name);
  CHECK(w_tensor->persistable());
  std::vector<float> w_scales;
  auto has_w_scale = op->HasInputScale(w_scale_name, true);
  if (has_w_scale) {
    w_scales = op->GetInputScale(w_scale_name, true);
  }
  auto w_precison = w_tensor->precision();
  auto w_dims = w_tensor->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  int64_t K = w_dims[0];
  int64_t N = w_dims[1];

  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }

  int in_num_col_dims = op->GetAttr<int>("in_num_col_dims");
  std::string act_type;
  if (op->HasAttr("activation_type")) {
    act_type = op->GetAttr<std::string>("activation_type");
  }

  // Check quantization mode
  bool is_quant_mode = false;
  if (w_precison == PRECISION(kInt8)) {
    CHECK(IsValidSymmQuantParams(w_scales))
        << "Missing the quant params '" << w_scale_name << "' for the input '"
        << w_name << "'";
    CHECK(IsValidSymmPerLayerQuantParams(input_scales))
        << "Missing the quant params '" << input_scale_name
        << "' for the input '" << input_name << "'";
    CHECK(IsValidSymmPerLayerQuantParams(out_scales))
        << "Missing the quant params '" << out_scale_name << "' for the out '"
        << out_name << "'";
    is_quant_mode = true;
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Reshape input operand
  std::vector<int32_t> shape;
  for (uint32_t i = 0; i < in_num_col_dims; i++) {
    shape.push_back(0);
  }
  shape.push_back(K);
  auto shape_operand = converter->AddConstantOperand(shape);
  auto input_reshape_operand =
      converter->AddOutputOperand(out_name, input_scales);
  converter->AddOperation(NNADAPTER_RESHAPE,
                          {input_operand, shape_operand},
                          {input_reshape_operand});
  // Weight operand
  NNAdapterOperand* weight_operand = nullptr;
  std::vector<float> bias_scales;
  if (is_quant_mode) {
    CHECK(IsNNInt8SymmPerLayerQuantType(*input_type));
    std::vector<float> quant_scales;
    CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
    CHECK(IsSameSymmQuantParams(input_scales, quant_scales));
    if (!IsValidSymmPerChannelQuantParams(w_scales)) {
      w_scales = {w_scales[0]};
    }
    weight_operand =
        converter->AddConstantOperand(*w_tensor, {}, false, w_scales);
    bias_scales.resize(w_scales.size());
    for (size_t i = 0; i < w_scales.size(); i++) {
      bias_scales[i] = input_scales[0] * w_scales[i];
    }
  } else {
    CHECK(input_type->precision ==
          ConvertPrecisionTypeToNNPrecisionCode(w_precison));
    weight_operand = converter->AddConstantOperand(*w_tensor);
  }
  // Transpose_x operand
  auto transpose_x_operand = converter->AddConstantOperand(false);
  // Transpose_y operand
  auto transpose_y_operand = converter->AddConstantOperand(false);
  // Output operand
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Matmul operation
  converter->AddOperation(NNADAPTER_MAT_MUL,
                          {input_reshape_operand,
                           weight_operand,
                           transpose_x_operand,
                           transpose_y_operand},
                          {output_operand});

  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  if (HasInput(op, scope, "Bias")) {
    auto bias_name = op->Input("Bias").front();
    bias_operand = converter->GetMappedOperand(bias_name);
    if (!bias_operand) {
      auto bias_tensor = scope->FindMutableTensor(bias_name);
      CHECK(bias_tensor->persistable());
      auto bias_precison = bias_tensor->precision();
      auto bias_dims = bias_tensor->dims();
      if (bias_dims.production() != N) {
        LOG(FATAL)
            << "Only supports bias_dims.production() == weight_dims[1] !";
        return UNSUPPORTED_FEATURE;
      }
      if (is_quant_mode) {
        CHECK(bias_tensor->precision() == PRECISION(kFloat));
        auto bias_data = bias_tensor->mutable_data<float>();
        std::vector<int32_t> quantized_bias_data(N, 0);
        SymmQuantizeData(bias_data, N, bias_scales, &quantized_bias_data[0]);
        bias_operand = converter->AddConstantOperand(
            quantized_bias_data, DDim({N}), bias_scales);
      } else {
        CHECK(input_type->precision ==
              ConvertPrecisionTypeToNNPrecisionCode(bias_precison));
        bias_operand = converter->AddConstantOperand(*bias_tensor, DDim({N}));
      }
    } else {
      auto bias_type = converter->GetOperandType(bias_operand);
      // Check if we can use the bias_operand directly
      if (is_quant_mode) {
        CHECK(IsNNInt32SymmQuantType(*bias_type));
        std::vector<float> quant_scales;
        CHECK(GetNNSymmQuantParams(*bias_type, &quant_scales));
        CHECK(IsSameSymmQuantParams(bias_scales, quant_scales));
      } else {
        CHECK(bias_type->precision == input_type->precision);
      }
    }
  } else {
    // Add dummy zero bias operand
    // Use int32 as the data type of bias if it is a quantized type
    std::vector<int32_t> zeros(
        N * (is_quant_mode ? sizeof(int32_t)
                           : GetNNOperandPrecisionDataLength(*input_type)),
        0);
    bias_operand = converter->AddConstantOperand(
        reinterpret_cast<void*>(zeros.data()),
        DDim({N}),
        is_quant_mode ? NNADAPTER_INT32 : input_type->precision,
        true,
        bias_scales);
  }
  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = NNADAPTER_FUSED_RELU;
    act_type = "";
  } else if (act_type == "relu1") {
    fuse_code_value = NNADAPTER_FUSED_RELU1;
    act_type = "";
  } else if (act_type == "relu6") {
    fuse_code_value = NNADAPTER_FUSED_RELU6;
    act_type = "";
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
  // Add operation to add bias to the result of Matmul operation
  auto eltwise_add_output_operand =
      converter->AddOutputOperand(out_name, out_scales);
  converter->AddOperation(NNADAPTER_ADD,
                          {output_operand, bias_operand, fuse_code_operand},
                          {eltwise_add_output_operand});
  // Unpack the fused activations
  converter->UnpackFusedActivations(
      eltwise_add_output_operand, act_type, op, scope, out_name, out_scales);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
