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

#include "lite/kernels/nnadapter/converter/converter.h"
#include <memory>
#include <utility>
#include "lite/kernels/nnadapter/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

#define REGISTER_CONVERTER(__op_type__, __func_name__, ...) \
  extern int __func_name__(Converter* converter, OpInfo* op, Scope* scope);
#include "lite/kernels/nnadapter/converter/all.h"  // NOLINT
#undef __NNADAPTER_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(int block_idx,
                     const cpp::ProgramDesc* program_desc,
                     Scope* exec_scope,
                     const std::vector<Variable>& input_vars,
                     std::vector<Variable>* output_vars,
                     std::vector<NNAdapterOperand*>* input_operands,
                     std::vector<NNAdapterOperand*>* output_operands) {
  CHECK(program_desc);
  CHECK(exec_scope);
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size) << "No block found!";
  CHECK(block_idx >= 0 && block_idx < block_size)
      << "Invalid block index, expected [0," << (block_size - 1)
      << "] but recieved " << block_idx;
  auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  auto op_size = block_desc->OpsSize();
  CHECK(op_size) << "No op found!";
  auto input_count = input_vars.size();
  input_operands->resize(input_count);
  // Prepare the input operands from the input tensors of the current subgraph
  for (size_t i = 0; i < input_count; i++) {
    const auto& name = input_vars[i].name;
    auto value = input_vars[i].value;
    CHECK(!value->persistable());  // Should not be a constant tensor
    auto quant_scale = input_vars[i].quant_scale;
    auto quant_zero_point = input_vars[i].quant_zero_point;
    CHECK_EQ(quant_zero_point, 0);
    // Get the dimensions of the input tensors, and set the specified axis to
    // NNADAPTER_UNKNOWN according to the dynamic dimension information
    auto dimensions = value->dims();
    auto dimension_count = dimensions.size();
    auto precision_type = value->precision();
    const auto& dynamic_dimensions = input_vars[i].dynamic_dimensions;
    auto dynamic_dimension_count = dynamic_dimensions.size();
    if (dynamic_dimension_count > 0) {
      CHECK_GE(dynamic_dimension_count, 2)
          << "The gear count of dynamic dimensions should be greater or equal "
             "to 2, but recieved "
          << dynamic_dimension_count;
      // Verify the dynamic dimensions
      std::vector<size_t> dynamic_axes;
      for (size_t k = 0; k < dimension_count; k++) {
        for (size_t j = 0; j < dynamic_dimension_count; j++) {
          CHECK_EQ(dynamic_dimensions[j].size(), dimension_count)
              << "The dimension count of " << j << " th dynamic dimensions of "
              << i << "th input '" << name << "' shoud be " << dimension_count
              << ", but recieved " << dynamic_dimensions[j].size();
          if (dynamic_dimensions[j][k] != dimensions[k]) {
            dynamic_axes.push_back(k);
            break;
          }
        }
      }
      CHECK_GT(dynamic_axes.size(), 0)
          << "Unable to find a dynamic axis from the dynamic dimensions of "
          << i << "th input '" << name << "'";
      for (size_t dynamic_axis : dynamic_axes) {
        dimensions[dynamic_axis] = NNADAPTER_UNKNOWN;
      }
    }
    // Is a quantization variable
    std::vector<float> quant_scales;
    if (IsValidSymmQuantParams({quant_scale})) {
      quant_scales.emplace_back(quant_scale);
    }
    auto operand = AddInputOperand(
        name, dimensions, dynamic_dimensions, precision_type, quant_scales);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand) << " for "
            << i << "th input '" << name << "'.";
    (*input_operands)[i] = operand;
  }
  // Convert the Paddle ops to some API calls of NNAdapter model builder
  for (size_t op_idx = 0; op_idx < op_size; op_idx++) {
    auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    auto op_info = std::make_shared<OpInfo>(*op_desc);
    VLOG(5) << "Converting " << op_type << " ...";
#define REGISTER_CONVERTER(__op_type__, __func_name__, ...) \
  if (op_type == #__op_type__) {                            \
    __func_name__(this, op_info.get(), exec_scope);         \
    continue;                                               \
  }
#include "lite/kernels/nnadapter/converter/all.h"  // NOLINT
#undef __NNADAPTER_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
    LOG(FATAL) << "Unsupported type '" << op_type << "' of " << op_idx
               << "th op in block " << block_idx;
  }
  // Query the output operands, and update if exists the useless output
  // variables such as 'XShape' in reshape2 and transpose2
  std::vector<Variable> valid_output_vars;
  auto output_count = output_vars->size();
  output_operands->clear();
  for (size_t i = 0; i < output_count; i++) {
    const auto& name = output_vars->at(i).name;
    auto operand = GetMappedOperand(name);
    if (!operand) {
      LOG(WARNING) << "No operand found for " << i << "th output '" << name
                   << "'!";
      continue;
    }
    output_operands->push_back(operand);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand) << " for "
            << i << "th output '" << name << "'.";
    valid_output_vars.emplace_back(output_vars->at(i));
  }
  CHECK_GT(valid_output_vars.size(), 0);
  if (valid_output_vars.size() != output_count) {
    *output_vars = valid_output_vars;
  }
  return NO_ERROR;
}

NNAdapterOperand* Converter::GetMappedOperand(const std::string& name) {
  auto it = operands_.find(name);
  if (it != operands_.end()) {
    return it->second.back();
  }
  return nullptr;
}

NNAdapterOperand* Converter::UpdateOperandMap(const std::string& name,
                                              NNAdapterOperand* operand) {
  auto it = operands_.find(name);
  if (it == operands_.end()) {
    auto result = operands_.insert(
        std::make_pair(name, std::vector<NNAdapterOperand*>()));
    CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(operand);
  return operand;
}

NNAdapterOperand* Converter::AddConstantOperand(
    const Tensor& tensor,
    DDim dimensions,
    bool copy,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  CHECK(tensor.persistable());
  if (dimensions.empty()) {
    dimensions = tensor.dims();
  } else {
    CHECK_EQ(tensor.dims().production(), dimensions.production());
  }
  auto precision_type = tensor.precision();
  auto buffer = tensor.raw_data();
  return AddOperand(dimensions,
                    precision_type,
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    buffer,
                    copy);
}

NNAdapterOperand* Converter::AddConstantOperand(
    const void* buffer,
    const DDim& dimensions,
    PrecisionType precision_type,
    bool copy,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  return AddOperand(dimensions,
                    precision_type,
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    buffer,
                    copy);
}

NNAdapterOperand* Converter::AddConstantOperand(
    const void* buffer,
    const DDim& dimensions,
    NNAdapterOperandPrecisionCode precision_code,
    bool copy,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  auto precision_type = ConvertNNPrecisionCodeToPrecisionType(precision_code);
  return AddOperand(dimensions,
                    precision_type,
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    buffer,
                    copy);
}

NNAdapterOperand* Converter::AddInputOperand(
    const std::string& name,
    const DDim& dimensions,
    const std::vector<std::vector<int64_t>>& dynamic_dimensions,
    PrecisionType precision_type,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  return AddOperand(dimensions,
                    precision_type,
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    nullptr,
                    false,
                    name,
                    dynamic_dimensions);
}

NNAdapterOperand* Converter::AddInputOperand(
    const std::string& name,
    const DDim& dimensions,
    const std::vector<std::vector<int64_t>>& dynamic_dimensions,
    NNAdapterOperandPrecisionCode precision_code,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  auto precision_type = ConvertNNPrecisionCodeToPrecisionType(precision_code);
  return AddOperand(dimensions,
                    precision_type,
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    nullptr,
                    false,
                    name,
                    dynamic_dimensions);
}

NNAdapterOperand* Converter::AddOutputOperand(
    const std::string& name,
    const std::vector<float>& quant_scales,
    uint32_t quant_channel_dim) {
  return AddOperand({},
                    PRECISION(kUnk),
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    nullptr,
                    false,
                    name);
}

NNAdapterOperand* Converter::AddOutputOperand(
    const std::vector<float>& quant_scales, uint32_t quant_channel_dim) {
  return AddOperand({},
                    PRECISION(kUnk),
                    quant_scales.data(),
                    quant_scales.size(),
                    quant_channel_dim,
                    nullptr,
                    false);
}

NNAdapterOperandType* Converter::GetOperandType(NNAdapterOperand* operand) {
  NNAdapterOperandType* type = nullptr;
  NNAdapterModel_getOperandType_invoke(operand, &type);
  CHECK(type);
  return type;
}

NNAdapterOperation* Converter::AddOperation(
    NNAdapterOperationType type,
    std::vector<NNAdapterOperand*>* input_operands,
    std::vector<NNAdapterOperand*>* output_operands) {
  NNAdapterOperation* operation = nullptr;
  NNAdapterModel_addOperation_invoke(model_,
                                     type,
                                     input_operands->size(),
                                     input_operands->data(),
                                     output_operands->size(),
                                     output_operands->data(),
                                     &operation);
  CHECK(operation);
  return operation;
}

NNAdapterOperand* Converter::AddOperand(NNAdapterOperandType* type,
                                        const std::string& name) {
  NNAdapterOperand* operand = nullptr;
  NNAdapterModel_addOperand_invoke(model_, type, &operand);
  CHECK(operand);
  if (!name.empty()) {
    if (GetMappedOperand(name)) {
      LOG(WARNING) << "Operand '" << name << "' already exists!";
    }
    UpdateOperandMap(name, operand);
  } else {
    // Anonymous operand
  }
  return operand;
}

NNAdapterOperand* Converter::AddOperand(
    const DDim& dimensions,
    PrecisionType precision_type,
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    const void* buffer,
    bool copy,
    const std::string& name,
    const std::vector<std::vector<int64_t>>& dynamic_dimensions) {
  NNAdapterOperandType type;
  memset(&type, 0, sizeof(NNAdapterOperandType));
  if (dimensions.size() > 0) {
    ConvertDDimToNNDimensions(
        dimensions, type.dimensions, &type.dimension_count);
  }
  type.dynamic_dimension_count = dynamic_dimensions.size();
  if (type.dynamic_dimension_count > 0) {
    for (uint32_t i = 0; i < type.dynamic_dimension_count; i++) {
      ConvertVectorToNNDimensions(dynamic_dimensions[i],
                                  type.dynamic_dimensions[i]);
    }
  }
  const auto UNKNOWN_PRECISION =
      static_cast<NNAdapterOperandPrecisionCode>(NNADAPTER_UNKNOWN);
  type.precision =
      precision_type != PRECISION(kUnk)
          ? ConvertPrecisionTypeToNNPrecisionCode(precision_type,
                                                  quant_scales,
                                                  quant_scale_count,
                                                  quant_channel_dim)
          : UNKNOWN_PRECISION;
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      CHECK(type.precision == UNKNOWN_PRECISION ||
            IsNNSymmPerChannelQuantType(type.precision));
      type.symm_per_channel_params.scales = const_cast<float*>(quant_scales);
      type.symm_per_channel_params.scale_count = quant_scale_count;
      type.symm_per_channel_params.channel_dim = quant_channel_dim;
    } else {
      // Symmetric per-layer quantization
      CHECK(type.precision == UNKNOWN_PRECISION ||
            IsNNSymmPerLayerQuantType(type.precision));
      type.symm_per_layer_params.scale = quant_scales[0];
    }
  } else {
    // Basic type, without any quantization parameters
  }
  auto operand = AddOperand(&type, name);
  if (buffer) {
    // Constant operand
    CHECK(type.precision != UNKNOWN_PRECISION);
    auto length = GetNNOperandTypeBufferLength(type);
    SetOperandValue(operand, buffer, length, copy);
  } else {
    // Variable/Input/Output operand
  }
  return operand;
}

void Converter::SetOperandValue(NNAdapterOperand* operand,
                                const void* buffer,
                                size_t length,
                                bool copy) {
  NNAdapterModel_setOperandValue_invoke(
      operand, const_cast<void*>(buffer), length, copy);
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
