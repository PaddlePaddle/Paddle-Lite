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

#include "nnadapter_optimizer.h"  // NOLINT
#include <memory>
#include <vector>
#include "nnadapter_common.h"   // NOLINT
#include "nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {

static void ConvertOperandFromSymmToAsymm(driver::Operand* operand,
                                          int32_t zero_point) {
  switch (operand->type.precision) {
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER: {
      operand->type.precision = NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER;
      auto scale = operand->type.symm_per_layer_params.scale;
      operand->type.asymm_per_layer_params = {.scale = scale,
                                              .zero_point = zero_point};
      auto is_constant_copy = operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
      auto is_constant_reference =
          operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
      if (zero_point != 0 && (is_constant_copy || is_constant_reference)) {
        auto transform_buffer = static_cast<uint8_t*>(operand->buffer);
        if (is_constant_reference) {
          transform_buffer = static_cast<uint8_t*>(malloc(operand->length));
        }
        for (uint32_t i = 0; i < operand->length; i++) {
          transform_buffer[i] = static_cast<uint8_t>(
              static_cast<int16_t>(static_cast<int8_t*>(operand->buffer)[i]) +
              zero_point);
        }
        if (is_constant_reference) {
          operand->buffer = transform_buffer;
          operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
        }
      }
    } break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL: {
    } break;
    default:
      break;
  }
}

NNADAPTER_EXPORT void ConvertQuantizationFromSymmToAsymm(driver::Model* model) {
  std::vector<Operation*> operations =
      driver::SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    switch (operation->type) {
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        // The zeroPoint of the output of softmax and sigmoid must be 0.
        ConvertOperandFromSymmToAsymm(output_operands[0], 0);
      } break;
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(input_operands[1], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_CONV_2D: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(input_operands[1], 128);
        ConvertOperandFromSymmToAsymm(input_operands[2], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      default:
        NNADAPTER_LOG(ERROR)
            << "Missing the processing of "
            << OperationTypeToString(operation->type)
            << " for the conversion of symm2asymm quantization.";
        break;
    }
  }
}

static void InsertNCHWToNHWCConversionToInputsOutputs(driver::Model* model) {
  for (auto& model_input_operand : model->input_operands) {
    if (model_input_operand->type.dimension_count == 4) {
      // Insert a transpose operation at the front of the model input operand to
      // convert NCHW to NHWC
      std::vector<int32_t> permutation = {0, 2, 3, 1};
      auto perm_operand = AddInt32ConstantOperand(model, permutation);
      auto input_operand = AddOperand(model);
      memcpy(&input_operand->type,
             &model_input_operand->type,
             sizeof(NNAdapterOperandType));
      std::vector<int32_t> inv_permutation = {0, 3, 1, 2};
      TransposeDimensions(input_operand->type.dimensions, inv_permutation);
      input_operand->type.lifetime = NNADAPTER_MODEL_INPUT;
      input_operand->type.layout = NNADAPTER_NCHW;
      auto transpose_operation = AddOperation(model);
      transpose_operation->type = NNADAPTER_TRANSPOSE;
      transpose_operation->input_operands.push_back(input_operand);
      transpose_operation->input_operands.push_back(perm_operand);
      transpose_operation->output_operands.push_back(model_input_operand);
      model_input_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
      model_input_operand = input_operand;
    }
  }
  for (auto& model_output_operand : model->output_operands) {
    if (model_output_operand->type.dimension_count == 4) {
      // Insert a transpose operation at the back of the model output operand to
      // convert NHWC to NCHW
      std::vector<int32_t> permutation = {0, 3, 1, 2};
      auto perm_operand = AddInt32ConstantOperand(model, permutation);
      auto output_operand = AddOperand(model);
      memcpy(&output_operand->type,
             &model_output_operand->type,
             sizeof(NNAdapterOperandType));
      TransposeDimensions(output_operand->type.dimensions, permutation);
      output_operand->type.lifetime = NNADAPTER_MODEL_OUTPUT;
      output_operand->type.layout = NNADAPTER_NCHW;
      auto transpose_operation = AddOperation(model);
      transpose_operation->type = NNADAPTER_TRANSPOSE;
      transpose_operation->input_operands.push_back(model_output_operand);
      transpose_operation->input_operands.push_back(perm_operand);
      transpose_operation->output_operands.push_back(output_operand);
      model_output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
      model_output_operand = output_operand;
    }
  }
}

inline bool IsOperandConvertToNHWC(driver::Operand* operand) {
  return operand->type.layout == NNADAPTER_NHWC;
}

inline bool MarkOperandConvertToNHWC(driver::Operand* operand) {
  return operand->type.layout = NNADAPTER_NHWC;
}

static void ConvertAverageAndMaxPool2DFromNCHWToNHWC(
    driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 12);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  auto output_operand = output_operands[0];
  switch (input_dimension_count) {
    case 4: {
      if (!IsOperandConvertToNHWC(input_operand)) {
        TransposeOperand(input_operand, {0, 2, 3, 1});
        MarkOperandConvertToNHWC(input_operand);
      }
      TransposeOperand(output_operand, {0, 2, 3, 1});
      MarkOperandConvertToNHWC(output_operand);
    } break;
    default:
      NNADAPTER_LOG(ERROR) << "Unhandled case: dimension_count="
                           << input_dimension_count;
      break;
  }
}

static int ConvertActivationUnaryOperationsFromNCHWToNHWC(
    driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  auto output_operand = output_operands[0];
  switch (input_dimension_count) {
    case 4: {
      if (!IsOperandConvertToNHWC(input_operand)) {
        TransposeOperand(input_operand, {0, 2, 3, 1});
        MarkOperandConvertToNHWC(input_operand);
      }
      TransposeOperand(output_operand, {0, 2, 3, 1});
      MarkOperandConvertToNHWC(output_operand);
    } break;
    default:
      NNADAPTER_LOG(ERROR) << "Unhandled case: dimension_count="
                           << input_dimension_count;
      break;
  }
}

static void ConvertElementwiseBinaryOperationsFromNCHWToNHWC(
    driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input0_operand = input_operands[0];
  auto input0_dimension_count = input0_operand->type.dimension_count;
  NNADAPTER_CHECK_GT(input0_dimension_count, 0);
  auto input1_operand = input_operands[1];
  auto input1_dimension_count = input1_operand->type.dimension_count;
  NNADAPTER_CHECK_GT(input1_dimension_count, 0);
  auto output_operand = output_operands[0];
  switch (input0_dimension_count) {
    case 4: {
      if (!IsOperandConvertToNHWC(input0_operand)) {
        TransposeOperand(input0_operand, {0, 2, 3, 1});
        MarkOperandConvertToNHWC(input0_operand);
      }
      if (!IsOperandConvertToNHWC(input1_operand)) {
        if (input1_dimension_count == 2) {
          // Pad 1 to the tail of the dimensions and transpose it as input0
          ReshapeOperand(input1_operand, {0, 0, 1});
        } else if (input1_dimension_count == 3) {
          TransposeOperand(input1_operand, {1, 2, 0});
        } else if (input1_dimension_count == 4) {
          TransposeOperand(input1_operand, {0, 2, 3, 1});
        }
        MarkOperandConvertToNHWC(input1_operand);
      }
      TransposeOperand(output_operand, {0, 2, 3, 1});
      MarkOperandConvertToNHWC(output_operand);
    } break;
    default:
      NNADAPTER_LOG(ERROR) << "Unhandled case: dimension_count="
                           << input0_dimension_count;
      break;
  }
}

static void ConvertSoftmaxFromNCHWToNHWC(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  auto output_operand = output_operands[0];
  switch (input_dimension_count) {
    case 2: {
      if (!IsOperandConvertToNHWC(input_operand)) {
        MarkOperandConvertToNHWC(input_operand);
      }
      MarkOperandConvertToNHWC(output_operand);
    } break;
    default:
      NNADAPTER_LOG(ERROR) << "Unhandled case: dimension_count="
                           << input_dimension_count;
      break;
  }
}

static void ConvertFullyConnectedFromNCHWToNHWC(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimension_count = input_operand->type.dimension_count;
  auto weight_operand = input_operands[1];
  auto bias_operand = input_operands[2];
  auto output_operand = output_operands[0];
  switch (input_dimension_count) {
    case 4: {
      if (!IsOperandConvertToNHWC(input_operand)) {
        TransposeOperand(input_operand, {0, 2, 3, 1});
        MarkOperandConvertToNHWC(input_operand);
      }
      MarkOperandConvertToNHWC(weight_operand);
      MarkOperandConvertToNHWC(bias_operand);
      MarkOperandConvertToNHWC(output_operand);
    } break;
    default:
      NNADAPTER_LOG(ERROR) << "Unhandled case: dimension_count="
                           << input_dimension_count;
      break;
  }
}

static void ConvertConv2DFromNCHWToNHWC(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimension_count = input_operand->type.dimension_count;
  NNADAPTER_CHECK_EQ(input_dimension_count, 4);
  if (!IsOperandConvertToNHWC(input_operand)) {
    TransposeOperand(input_operand, {0, 2, 3, 1});
    MarkOperandConvertToNHWC(input_operand);
  }
  auto input_channel_size = input_operand->type.dimensions[3];
  auto filter_operand = input_operands[1];
  bool is_per_channel = filter_operand->type.precision ==
                        NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto group = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  // Check depthwise mode
  bool is_depthwise_mode = input_channel_size == group;
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
  std::vector<int32_t> filter_permutation = {};
  if (is_per_channel) {
    filter_operand->type.symm_per_channel_params.channel_dim =
        is_depthwise_mode ? 3 : 0;
  }
  if (is_depthwise_mode) {
    // [C_out, 1, filter_height, filter_width]->[1, filter_height, filter_width,
    // C_out]
    filter_permutation = {1, 2, 3, 0};
  } else {
    // [C_out, C_in, filter_height, filter_width]->[C_out, filter_height,
    // filter_width, C_in]
    filter_permutation = {0, 2, 3, 1};
  }
  TransposeOperand(filter_operand, filter_permutation);
  MarkOperandConvertToNHWC(filter_operand);
  auto bias_operand = input_operands[2];
  MarkOperandConvertToNHWC(bias_operand);
  auto output_operand = output_operands[0];
  TransposeOperand(output_operand, {0, 2, 3, 1});
  MarkOperandConvertToNHWC(output_operand);
}

NNADAPTER_EXPORT void ConvertDataLayoutFromNCHWToNHWC(driver::Model* model) {
  std::vector<Operation*> operations =
      driver::SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertAverageAndMaxPool2DFromNCHWToNHWC(operation);
        break;
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_SIGMOID:
        ConvertActivationUnaryOperationsFromNCHWToNHWC(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmaxFromNCHWToNHWC(operation);
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB:
        ConvertElementwiseBinaryOperationsFromNCHWToNHWC(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnectedFromNCHWToNHWC(operation);
        break;
        break;
      case NNADAPTER_CONV_2D:
        ConvertConv2DFromNCHWToNHWC(operation);
        break;
      default:
        NNADAPTER_LOG(ERROR) << "Missing the processing of "
                             << OperationTypeToString(operation->type)
                             << " for the conversion from NCHW to NHWC.";
        break;
    }
  }
  InsertNCHWToNHWCConversionToInputsOutputs(model);
}

}  // namespace driver
}  // namespace nnadapter
