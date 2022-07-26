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

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertScale(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  auto has_x_scale = op->HasInputScale(x_scale_name, true);
  if (has_x_scale) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_tensor = scope->FindTensor(x_name);
  auto input_precision = input_tensor->precision();
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Scale and bias
  auto scale = op->GetAttr<float>("scale");
  auto bias = op->GetAttr<float>("bias");
  auto bias_after_scale = op->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }
  auto has_scale = fabs(scale - 1.0f) > 1e-5f;
  auto has_bias = fabs(bias) > 1e-5f;

  // Output
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }

  if (!has_scale && !has_bias) {
    converter->UpdateOperandMap(out_name, input_operand);
  } else if (has_scale) {
    // Scale operand
    NNAdapterOperand* scale_operand = nullptr;
    if (has_x_scale) {
      int8_t quant_scale_data = scale > 0.0f ? 1 : -1;
      std::vector<int8_t> quant_scale_data_vec;
      quant_scale_data_vec.push_back(quant_scale_data);
      scale_operand = converter->AddConstantOperand(
          reinterpret_cast<void*>(quant_scale_data_vec.data()),
          DDim({static_cast<int64_t>(1)}),
          NNADAPTER_INT8,
          true,
          {fabsf(scale)});
    } else {
      if (input_precision == PrecisionType::kFloat) {
        scale_operand = converter->AddConstantOperand(scale);
      } else if (input_precision == PrecisionType::kInt32) {
        scale_operand =
            converter->AddConstantOperand(static_cast<int32_t>(scale));
      } else if (input_precision == PrecisionType::kInt64) {
        scale_operand =
            converter->AddConstantOperand(static_cast<int64_t>(scale));
      } else {
        LOG(FATAL) << "Unsupported a fluid data type("
                   << static_cast<int32_t>(input_precision)
                   << ") for scale operand";
      }
    }
    // Fuse code operand
    int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
    auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
    auto scale_output_operand =
        converter->AddOutputOperand(out_name, out_scales);
    // Mul operation for input*scale
    converter->AddOperation(NNADAPTER_MUL,
                            {input_operand, scale_operand, fuse_code_operand},
                            {scale_output_operand});
    // Bias operand
    NNAdapterOperand* bias_operand = nullptr;
    if (has_bias) {
      if (has_x_scale) {
        int8_t quant_bias_data = bias > 0.0f ? 1 : -1;
        std::vector<int8_t> quant_bias_data_vec;
        quant_bias_data_vec.push_back(quant_bias_data);
        bias_operand = converter->AddConstantOperand(
            reinterpret_cast<void*>(quant_bias_data_vec.data()),
            DDim({static_cast<int64_t>(1)}),
            NNADAPTER_INT8,
            true,
            {fabsf(bias)});
      } else {
        if (input_precision == PrecisionType::kFloat) {
          bias_operand = converter->AddConstantOperand(bias);
        } else if (input_precision == PrecisionType::kInt32) {
          bias_operand =
              converter->AddConstantOperand(static_cast<int32_t>(bias));
        } else if (input_precision == PrecisionType::kInt64) {
          bias_operand =
              converter->AddConstantOperand(static_cast<int64_t>(bias));
        } else {
          LOG(FATAL) << "Unsupported a fluid data type("
                     << static_cast<int32_t>(input_precision)
                     << ") for bias operand";
        }
      }
      // Fuse code operand
      auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
      auto bias_output_operand =
          converter->AddOutputOperand(out_name, out_scales);
      // Add operation for input+bias or input*scale+bias
      converter->AddOperation(
          NNADAPTER_ADD,
          {scale_output_operand, bias_operand, fuse_code_operand},
          {bias_output_operand});
    }
  } else {
    // Has bias only
    NNAdapterOperand* bias_operand = nullptr;
    if (has_x_scale) {
      int8_t quant_bias_data = bias > 0.0f ? 1 : -1;
      std::vector<int8_t> quant_bias_data_vec;
      quant_bias_data_vec.push_back(quant_bias_data);
      bias_operand = converter->AddConstantOperand(
          reinterpret_cast<void*>(quant_bias_data_vec.data()),
          DDim({static_cast<int64_t>(1)}),
          NNADAPTER_INT8,
          true,
          {fabsf(bias)});
    } else {
      if (input_precision == PrecisionType::kFloat) {
        bias_operand = converter->AddConstantOperand(bias);
      } else if (input_precision == PrecisionType::kInt32) {
        bias_operand =
            converter->AddConstantOperand(static_cast<int32_t>(bias));
      } else if (input_precision == PrecisionType::kInt64) {
        bias_operand =
            converter->AddConstantOperand(static_cast<int64_t>(bias));
      } else {
        LOG(FATAL) << "Unsupported a fluid data type("
                   << static_cast<int32_t>(input_precision)
                   << ") for bias operand";
      }
    }
    // Fuse code operand
    int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
    auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
    auto output_operand = converter->AddOutputOperand(out_name, out_scales);
    // Add operation for input+bias or input*scale+bias
    converter->AddOperation(NNADAPTER_ADD,
                            {input_operand, bias_operand, fuse_code_operand},
                            {output_operand});
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
