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
  int64_t K = w_dims[0] int64_t N = w_dims[1];

  auto output_name = op->Output("Output").front();
  auto output_scale_name = "Output0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }

  int in_num_col_dims = op->GetAttr<int>("in_num_col_dims");
  std::string activation_type;
  if (op->HasAttr("activation_type")) {
    activation_type = op->GetAttr<std::string>("activation_type");
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
    CHECK(IsValidSymmPerLayerQuantParams(output_scales))
        << "Missing the quant params '" << output_scale_name
        << "' for the output '" << output_name << "'";
    is_quant_mode = true;
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Weight operand
  // Transpose to [k, n] to [n, k]
  NNAdapterOperand* weight_operand = nullptr;
  std::vector<float> bias_scales;
  if (is_quant_mode) {
    if (IsNNInt8SymmPerLayerQuantType(*input_type)) {
      std::vector<float> quant_scales;
      CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
      CHECK(IsSameSymmQuantParams(input_scales, quant_scales));
      // TODO(shentanyue) Add a NNADAPTER_DEQUANT&NNADAPTER_QUANT operation to
      // make the quant params obtained from a operand consistent with those
      // obtained from op_desc
    } else {
      // TODO(shentanyue) Add a NNADAPTER_QUANT/NNADAPTER_DEQUANT operation to
      // convert any type to int8 symm per-layer quant operand
      LOG(FATAL) << "Mixed precision will be supported in future!";
      return UNSUPPORTED_FEATURE;
    }
    if (!IsValidSymmPerChannelQuantParams(w_scales)) {
      w_scales = {w_scales[0]};
    }
    auto w_data = w_tensor->mutable_data<int8_t>();
    std::vector<int8_t> transpose_weight_data(w_dims.production(), 0);
    DDim transpose_w_dims;
    Transpose(
        w_data, &transpose_weight_data[0], {1, 0}, w_dims, &transpose_w_dims);
    weight_operand = converter->AddConstantOperand(
        transpose_weight_data, transpose_w_dims, w_scales);
    bias_scales.resize(w_scales.size());
    for (size_t i = 0; i < w_scales.size(); i++) {
      bias_scales[i] = input_scales[0] * w_scales[i];
    }
  } else {
    if (IsNNInt8SymmPerLayerQuantType(*input_type)) {
      // TODO(shentanyue) Add a NNADAPTER_DEQUANT to dequantize the input
      // operand to the same type of operand as the weight operand
      LOG(FATAL) << "Mixed precision will be supported in future!";
      return UNSUPPORTED_FEATURE;
    }
    CHECK(input_type->precision ==
          ConvertPrecisionTypeToNNPrecisionCode(w_precison));
    auto w_data = w_tensor->mutable_data<float>();
    std::vector<float> transpose_weight_data(w_dims.production(), 0);
    DDim transpose_w_dims;
    Transpose(
        w_data, &transpose_weight_data[0], {1, 0}, w_dims, &transpose_w_dims);
    weight_operand =
        converter->AddConstantOperand(transpose_weight_data, transpose_w_dims)
  }

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
      if (bias_dims.production() != K) {
        LOG(FATAL)
            << "Only supports bias_dims.production() == weight_dims[0] !";
        return UNSUPPORTED_FEATURE;
      }
      if (is_quant_mode) {
        CHECK(bias_tensor->precision() == PRECISION(kFloat));
        auto bias_data = bias_tensor->mutable_data<float>();
        std::vector<int32_t> quantized_bias_data(K, 0);
        SymmQuantizeData(bias_data, K, bias_scales, &quantized_bias_data[0]);
        bias_operand = converter->AddConstantOperand(
            quantized_bias_data, DDim({K}), bias_scales);
      } else {
        CHECK(input_type->precision ==
              ConvertPrecisionTypeToNNPrecisionCode(bias_precison));
        bias_operand = converter->AddConstantOperand(*bias_tensor, DDim({K}));
      }
    } else {
      auto bias_type = converter->GetOperandType(bias_operand);
      // Check if we can use the bias_operand directly
      if (is_quant_mode) {
        if (IsNNInt32SymmQuantType(*bias_type)) {
          std::vector<float> quant_scales;
          CHECK(GetNNSymmQuantParams(*bias_type, &quant_scales));
          CHECK(IsSameSymmQuantParams(bias_scales, quant_scales));
          // TODO(shentanyue) Add a NNADAPTER_DEQUANT&NNADAPTER_QUANT
          // operation to make the quant params obtained from a operand
          // consistent with those obtained from op_desc
        } else {
          // TODO(shentanyue) Add a NNADAPTER_QUANT/NNADAPTER_DEQUANT
          // operation to convert any type to int32 symm per-layer/per-channel
          // quant operand
          LOG(FATAL) << "Mixed precision will be supported in future!";
          return UNSUPPORTED_FEATURE;
        }
      } else {
        CHECK(bias_type->precision == input_type->precision);
      }
    }
  } else {
    // Add dummy zero bias operand
    // Use int32 as the data type of bias if it is a quantized type
    std::vector<int8_t> zeros(
        K * (is_quant_mode ? sizeof(int32_t)
                           : GetNNOperandPrecisionDataLength(*input_type)),
        0);
    bias_operand = converter->AddConstantOperand(
        reinterpret_cast<void*>(zeros.data()),
        DDim({K}),
        is_quant_mode ? NNADAPTER_INT32 : input_type->precision,
        true,
        bias_scales);
  }

  // Fuse code operand
  NNAdapterFuseCode fuse_code_value = NNADAPTER_FUSED_NONE;
  std::vector<std::string> activation_support_split_ops{
      "sigmoid", "tan", "log", "abs"};
  NNAdapterFuseCode fuse_code = NNADAPTER_FUSED_NONE;
  if (activation_type == "relu") {
    fuse_code = NNADAPTER_FUSED_RELU;
    activation_type = "";
  } else if (activation_type == "relu1") {
    fuse_code = NNADAPTER_FUSED_RELU1;
    activation_type = "";
  } else if (activation_type == "relu6") {
    fuse_code = NNADAPTER_FUSED_RELU6;
    activation_type = "";
  } else if (!activation_type.empty()) {
    if (std::find(activation_support_split_ops.begin(),
                  activation_support_split_ops.end(),
                  activation_type) == activation_support_split_ops.end()) {
      LOG(FATAL) << "NNadapter doesn't supported activation type : "
                 << activation_type << " fusion!";
      return UNSUPPORTED_FEATURE;
    }
    VLOG(5) << "Split fc + " << activation_type
            << " fusion operator into two operators!";
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
  // Output operand
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // Fully connected operation
  converter->AddOperation(
      NNADAPTER_FULLY_CONNECTED,
      {input_operand, weight_operand, bias_operand, fuse_code_operand},
      {output_operand});
  // Activation
  if (!activation_type.empty()) {
    auto activation_operand = converter->AddOutputOperand(output_name);
    NNAdapterOperationType act_type;
    if (activation_type == "sigmoid") {
      act_type = NNADAPTER_SIGMOID;
    } else if (activation_type == "tanh") {
      act_type = NNADAPTER_TANH;
    } else if (activation_type == "log") {
      act_type = NNADAPTER_LOG;
    } else if (activation_type == "abs") {
      act_type = NNADAPTER_ABS;
    } else {
      LOG(FATAL) << "Unsupported unary activation type: " << activation_type;
      return UNSUPPORTED_FEATURE;
    }
    converter->AddOperation(act_type, {output_operand}, {activation_operand});
    output_operand = activation_operand;
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
