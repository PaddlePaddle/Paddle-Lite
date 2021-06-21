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

#include "utility/modeling.h"
#include <algorithm>
#include <map>
#include <utility>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT hal::Operand* AddOperand(hal::Model* model) {
  model->operands.emplace_back();
  return &model->operands.back();
}

NNADAPTER_EXPORT hal::Operation* AddOperation(hal::Model* model) {
  model->operations.emplace_back();
  return &model->operations.back();
}

static hal::Operand* AddOperand(hal::Model* model,
                                const std::vector<int32_t>& dimensions,
                                NNAdapterOperandPrecisionCode precision,
                                float* quant_scales = nullptr,
                                int32_t* zero_point = nullptr,
                                uint32_t quant_scale_count = 0,
                                uint32_t quant_channel_dim = 0,
                                void* buffer = nullptr,
                                bool copy = true) {
  auto operand = AddOperand(model);
  memset(&operand->type, 0, sizeof(NNAdapterOperandType));
  operand->type.dimension_count = dimensions.size();
  if (!dimensions.empty()) {
    memcpy(operand->type.dimensions,
           &dimensions[0],
           dimensions.size() * sizeof(int32_t));
  }
  operand->type.precision = precision;
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      NNADAPTER_CHECK(
          !zero_point &&
              precision == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
          precision == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL);
      operand->type.symm_per_channel_params.scales = quant_scales;
      operand->type.symm_per_channel_params.scale_count = quant_scale_count;
      operand->type.symm_per_channel_params.channel_dim = quant_channel_dim;
    } else {
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK(
            precision == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
            precision == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER);
        operand->type.asymm_per_layer_params.scale = quant_scales[0];
        operand->type.asymm_per_layer_params.zero_point = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK(
            precision == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER ||
            precision == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER);
        operand->type.symm_per_layer_params.scale = quant_scales[0];
      }
    }
  } else {
    // Basic type, without any quantization parameters
  }
  if (buffer) {
    // Constant operand
    operand->length =
        OperandPrecisionLength(precision) * ProductionOfDimensions(dimensions);
    if (copy) {
      operand->buffer = malloc(operand->length);
      NNADAPTER_CHECK(operand->buffer != nullptr)
          << "Failed to allocate " << operand->length
          << " bytes for the buffer of an operand, out of memory!";
      memcpy(operand->buffer, buffer, operand->length);
      operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    } else {
      operand->buffer = buffer;
      operand->type.lifetime = NNADAPTER_CONSTANT_REFERENCE;
    }
  } else {
    // Variable/Input/Output operand
    operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  return operand;
}

NNADAPTER_EXPORT hal::Operand* AddBool8ConstantOperand(hal::Model* model,
                                                       bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(
      model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &int8_value);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(hal::Model* model,
                                                       int32_t value) {
  return AddOperand(model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(hal::Model* model,
                                                         float value) {
  return AddOperand(
      model, {}, NNADAPTER_FLOAT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(
    hal::Model* model, std::vector<int32_t> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    &values[0]);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(
    hal::Model* model, std::vector<float> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    &values[0]);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(
    hal::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(
    hal::Model* model,
    float* values,
    const std::vector<int32_t>& dimensions,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
    int8_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
    int8_t* values,
    const std::vector<int32_t>& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
    uint8_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
    uint32_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT hal::Operand* AddQuant8VariableOperand(
    hal::Model* model,
    const std::vector<int32_t>& dimensions,
    float quant_scale) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT hal::Operand* AddQuant8VariableOperand(
    hal::Model* model,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32VariableOperand(
    hal::Model* model, const std::vector<int32_t>& dimensions) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT void ReshapeOperand(hal::Operand* operand,
                                     std::vector<int32_t> dimensions) {
  ReshapeDimensions(
      operand->type.dimensions, &operand->type.dimension_count, dimensions);
}

NNADAPTER_EXPORT void TransposeOperand(hal::Operand* operand,
                                       std::vector<int32_t> permutation) {
  auto is_constant_copy = operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference =
      operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  NNADAPTER_CHECK(!permutation.empty()) << "Permutation is empty!";
  NNADAPTER_CHECK_EQ(permutation.size(), operand->type.dimension_count)
      << "The rank of permutation and operand mismatch!";
  if (is_constant) {
#define OPERAND_TRANSPOSE_DATA(bytes, dtype)                          \
  case bytes: {                                                       \
    auto src_buffer = reinterpret_cast<dtype*>(origin_buffer);        \
    auto dst_buffer = reinterpret_cast<dtype*>(transform_buffer);     \
    TransposeData<dtype>(                                             \
        src_buffer, dst_buffer, permutation, dimensions, dimensions); \
  } break;
    auto origin_buffer = operand->buffer;
    auto transform_buffer = malloc(operand->length);
    NNADAPTER_CHECK(transform_buffer) << "Out of memory!";
    auto dimensions = operand->type.dimensions;
    int bytes = OperandPrecisionLength(operand->type.precision);
    switch (bytes) {
      OPERAND_TRANSPOSE_DATA(1, int8_t);
      OPERAND_TRANSPOSE_DATA(2, int16_t);
      OPERAND_TRANSPOSE_DATA(4, int32_t);
      OPERAND_TRANSPOSE_DATA(8, int64_t);
      default:
        NNADAPTER_LOG(ERROR)
            << "Missing the processing of "
            << OperandPrecisionCodeToString(operand->type.precision)
            << " for the transpose of operand.";
        break;
    }
    if (is_constant_reference) {
      operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    } else {
      // Free th origin buffer and replace it with the new one
      free(origin_buffer);
    }
    operand->buffer = transform_buffer;
#undef OPERAND_TRANSPOSE_DATA
  } else {
    // Only transpose the dimensions the non-constant operands
    TransposeDimensions(operand->type.dimensions, permutation);
  }
}

NNADAPTER_EXPORT bool ReplaceOperand(hal::Model* model,
                                     hal::Operand* pattern,
                                     hal::Operand* replace,
                                     bool remove) {
  bool found = false;
  // Replace if any operation use the 'pattern' as input or output.
  for (auto& operation : model->operations) {
    for (auto& operand : operation.input_operands) {
      if (operand == pattern) {
        operand = replace;
      }
    }
    for (auto& operand : operation.output_operands) {
      if (operand == pattern) {
        operand = replace;
      }
    }
  }
  // Replace if the 'pattern' is a model input or output operand
  if (pattern->type.lifetime == NNADAPTER_MODEL_INPUT) {
    replace->type.lifetime = NNADAPTER_MODEL_INPUT;
    for (auto& operand : model->input_operands) {
      if (operand == pattern) {
        operand = replace;
      }
    }
  } else if (pattern->type.lifetime == NNADAPTER_MODEL_OUTPUT) {
    replace->type.lifetime = NNADAPTER_MODEL_OUTPUT;
    for (auto& operand : model->output_operands) {
      if (operand == pattern) {
        operand = replace;
      }
    }
  }
  if (remove) {
    auto pos =
        std::find_if(model->operands.begin(),
                     model->operands.end(),
                     [&pattern](hal::Operand& o) { return &o == pattern; });
    NNADAPTER_CHECK(pos != model->operands.end());
    model->operands.erase(pos);
  } else {
    if (pattern->type.lifetime == NNADAPTER_MODEL_INPUT ||
        pattern->type.lifetime == NNADAPTER_MODEL_OUTPUT) {
      pattern->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
    }
  }
  return found;
}

NNADAPTER_EXPORT hal::Operand* AddTransposeOperation(
    hal::Model* model,
    hal::Operand* input_operand,
    std::vector<int32_t> permutation) {
  auto output_operand = AddOperand(model);
  memcpy(&output_operand->type,
         &input_operand->type,
         sizeof(NNAdapterOperandType));
  TransposeDimensions(output_operand->type.dimensions, permutation);
  // Update if input_operand is a model input operand
  for (auto& operation : model->operations) {
    for (auto& operand : operation.input_operands) {
      if (operand == input_operand) {
        operand = output_operand;
      }
    }
  }
  if (input_operand->type.lifetime == NNADAPTER_MODEL_OUTPUT) {
    for (auto& operand : model->output_operands) {
      if (operand == input_operand) {
        operand = output_operand;
      }
    }
    output_operand->type.lifetime = NNADAPTER_MODEL_OUTPUT;
    input_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  auto perm_operand = AddInt32ConstantOperand(model, permutation);
  auto transpose_operation = AddOperation(model);
  transpose_operation->type = NNADAPTER_TRANSPOSE;
  transpose_operation->input_operands = {input_operand, perm_operand};
  transpose_operation->output_operands = {output_operand};
  return output_operand;
}

NNADAPTER_EXPORT std::vector<hal::Operation*> SortOperationsInTopologicalOrder(
    hal::Model* model) {
  std::vector<hal::Operation*> operations;  // Operations in topological order
  std::vector<hal::Operation*> queue;
  // Use to find all of adjacent operations according to a given operand.
  std::multimap<hal::Operand*, hal::Operation*> map;
  // The counters of variable inputs for all of operations.
  std::map<hal::Operation*, uint32_t> counts;
  for (auto& operation : model->operations) {
    uint32_t count = 0;
    for (auto operand : operation.input_operands) {
      auto lifetime = operand->type.lifetime;
      if (lifetime == NNADAPTER_TEMPORARY_VARIABLE ||
          lifetime == NNADAPTER_MODEL_OUTPUT) {
        count++;
        map.insert(
            std::pair<hal::Operand*, hal::Operation*>(operand, &operation));
      }
    }
    if (count == 0) {
      // The operation which only depends the model inputs and constants
      queue.push_back(&operation);
    }
    counts[&operation] = count;
  }
  while (queue.size() > 0) {
    auto operation = queue.back();
    queue.pop_back();
    operations.push_back(operation);
    for (auto operand : operation->output_operands) {
      auto range = map.equal_range(operand);
      for (auto i = range.first; i != range.second; i++) {
        uint32_t& count = counts[i->second];
        if (--count == 0) {
          queue.push_back(i->second);
        }
      }
    }
  }
  return operations;
}

}  // namespace nnadapter
