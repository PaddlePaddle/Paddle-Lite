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

#include "nnadapter_common.h"  // NOLINT
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include "nnadapter_logging.h"  // NOLINT
#include "nnadapter_micros.h"   // NOLINT

namespace nnadapter {

NNADAPTER_EXPORT std::string string_format(const std::string fmt_str, ...) {
  // Reserve two times as much as the length of the fmt_str
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    // Wrap the plain char array into the unique_ptr
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

#define NNADAPTER_TYPE_TO_STRING(type) \
  case NNADAPTER_##type:               \
    name = #type;                      \
    break;

NNADAPTER_EXPORT std::string ResultCodeToString(NNAdapterResultCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(NO_ERROR);
    NNADAPTER_TYPE_TO_STRING(OUT_OF_MEMORY);
    NNADAPTER_TYPE_TO_STRING(INVALID_PARAMETER);
    NNADAPTER_TYPE_TO_STRING(DEVICE_NOT_FOUND);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandPrecisionCodeToString(
    NNAdapterOperandPrecisionCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(BOOL8);
    NNADAPTER_TYPE_TO_STRING(INT8);
    NNADAPTER_TYPE_TO_STRING(UINT8);
    NNADAPTER_TYPE_TO_STRING(INT16);
    NNADAPTER_TYPE_TO_STRING(UINT16);
    NNADAPTER_TYPE_TO_STRING(INT32);
    NNADAPTER_TYPE_TO_STRING(UINT32);
    NNADAPTER_TYPE_TO_STRING(INT64);
    NNADAPTER_TYPE_TO_STRING(UINT64);
    NNADAPTER_TYPE_TO_STRING(FLOAT16);
    NNADAPTER_TYPE_TO_STRING(FLOAT32);
    NNADAPTER_TYPE_TO_STRING(FLOAT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_BOOL8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT8_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT32_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandLayoutCodeToString(
    NNAdapterOperandLayoutCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(NCHW);
    NNADAPTER_TYPE_TO_STRING(NHWC);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandLifetimeCodeToString(
    NNAdapterOperandLifetimeCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(TEMPORARY_VARIABLE);
    NNADAPTER_TYPE_TO_STRING(CONSTANT_COPY);
    NNADAPTER_TYPE_TO_STRING(CONSTANT_REFERENCE);
    NNADAPTER_TYPE_TO_STRING(MODEL_INPUT);
    NNADAPTER_TYPE_TO_STRING(MODEL_OUTPUT);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperationTypeToString(
    NNAdapterOperationType type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(ADD);
    NNADAPTER_TYPE_TO_STRING(AVERAGE_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(CONV_2D);
    NNADAPTER_TYPE_TO_STRING(DIV);
    NNADAPTER_TYPE_TO_STRING(FULLY_CONNECTED);
    NNADAPTER_TYPE_TO_STRING(HARD_SIGMOID);
    NNADAPTER_TYPE_TO_STRING(HARD_SWISH);
    NNADAPTER_TYPE_TO_STRING(MAX_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(MUL);
    NNADAPTER_TYPE_TO_STRING(RELU);
    NNADAPTER_TYPE_TO_STRING(RELU6);
    NNADAPTER_TYPE_TO_STRING(SIGMOID);
    NNADAPTER_TYPE_TO_STRING(SOFTMAX);
    NNADAPTER_TYPE_TO_STRING(SUB);
    NNADAPTER_TYPE_TO_STRING(TANH);
    NNADAPTER_TYPE_TO_STRING(TRANSPOSE);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string FuseCodeToString(NNAdapterFuseCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(FUSED_NONE);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU1);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU6);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string DimensionsToString(const int32_t* dimensions,
                                                uint32_t dimension_count) {
  std::string text;
  if (dimension_count >= 1) {
    text = string_format("%d", dimensions[0]);
    for (uint32_t i = 1; i < dimension_count; i++) {
      text += string_format(",%d", dimensions[i]);
    }
  }
  return text;
}

NNADAPTER_EXPORT std::string DeviceCodeToString(NNAdapterDeviceCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(CPU);
    NNADAPTER_TYPE_TO_STRING(GPU);
    NNADAPTER_TYPE_TO_STRING(ACCELERATOR);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

#undef NNADAPTER_TYPE_TO_STRING

NNADAPTER_EXPORT int OperandPrecisionLength(
    NNAdapterOperandPrecisionCode type) {
#define NNADAPTER_PRECISION_LENGTH(type, bytes) \
  case NNADAPTER_##type:                        \
    return bytes;
  switch (type) {
    NNADAPTER_PRECISION_LENGTH(BOOL8, 1);
    NNADAPTER_PRECISION_LENGTH(INT8, 1);
    NNADAPTER_PRECISION_LENGTH(UINT8, 1);
    NNADAPTER_PRECISION_LENGTH(INT16, 2);
    NNADAPTER_PRECISION_LENGTH(UINT16, 2);
    NNADAPTER_PRECISION_LENGTH(INT32, 4);
    NNADAPTER_PRECISION_LENGTH(UINT32, 4);
    NNADAPTER_PRECISION_LENGTH(INT64, 8);
    NNADAPTER_PRECISION_LENGTH(UINT64, 8);
    NNADAPTER_PRECISION_LENGTH(FLOAT16, 2);
    NNADAPTER_PRECISION_LENGTH(FLOAT32, 4);
    NNADAPTER_PRECISION_LENGTH(FLOAT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_BOOL8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT8_SYMM_PER_LAYER, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT32_SYMM_PER_LAYER, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER, 4);
    default:
      NNADAPTER_LOG(ERROR) << "Failed to get the length of "
                           << OperandPrecisionCodeToString(type) << ".";
      break;
  }
#undef NNADAPTER_PRECISION_LENGTH
  return 0;
}

NNADAPTER_EXPORT std::string OperandPrecisionName(
    NNAdapterOperandPrecisionCode type) {
#define NNADAPTER_PRECISION_NAME(type, name) \
  case NNADAPTER_##type:                     \
    return #name;
  switch (type) {
    NNADAPTER_PRECISION_NAME(BOOL8, b);
    NNADAPTER_PRECISION_NAME(INT8, i8);
    NNADAPTER_PRECISION_NAME(UINT8, u8);
    NNADAPTER_PRECISION_NAME(INT16, i16);
    NNADAPTER_PRECISION_NAME(UINT16, u16);
    NNADAPTER_PRECISION_NAME(INT32, i32);
    NNADAPTER_PRECISION_NAME(UINT32, u32);
    NNADAPTER_PRECISION_NAME(INT64, i64);
    NNADAPTER_PRECISION_NAME(UINT64, u64);
    NNADAPTER_PRECISION_NAME(FLOAT16, f16);
    NNADAPTER_PRECISION_NAME(FLOAT32, f32);
    NNADAPTER_PRECISION_NAME(FLOAT64, f64);
    NNADAPTER_PRECISION_NAME(TENSOR_BOOL8, b);
    NNADAPTER_PRECISION_NAME(TENSOR_INT8, i8);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT8, u8);
    NNADAPTER_PRECISION_NAME(TENSOR_INT16, i16);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT16, u16);
    NNADAPTER_PRECISION_NAME(TENSOR_INT32, i32);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT32, u32);
    NNADAPTER_PRECISION_NAME(TENSOR_INT64, i64);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT64, u64);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT16, f16);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT32, f32);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT64, f16);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT8_SYMM_PER_LAYER, qi8sl);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, qi8sc);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER, qu8al);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT32_SYMM_PER_LAYER, qi32sl);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, qi32sc);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER, qu32al);
    default:
      NNADAPTER_LOG(ERROR) << "Failed to get the name of "
                           << OperandPrecisionCodeToString(type) << ".";
      break;
  }
#undef NNADAPTER_PRECISION_NAME
  return 0;
}

NNADAPTER_EXPORT bool IsPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsAsymmetricQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsSymmetricQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsAsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsAsymmetricQuantization(type) && IsPerLayerQuantization(type);
}

NNADAPTER_EXPORT bool IsSymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsSymmetricQuantization(type) && IsPerLayerQuantization(type);
}

NNADAPTER_EXPORT bool IsSymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsSymmetricQuantization(type) && IsPerChannelQuantization(type);
}

NNADAPTER_EXPORT bool IsUInt8AsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt8SymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt8SymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsUInt32AsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt32SymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt32SymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT int64_t ProductionOfDimensions(
    const int32_t* input_dimensions, uint32_t input_dimension_count) {
  int64_t production = 1;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GT(dimension, 0);
    production *= dimension;
  }
  return production;
}

NNADAPTER_EXPORT int64_t
ProductionOfDimensions(const std::vector<int32_t>& input_dimensions) {
  return !input_dimensions.empty()
             ? ProductionOfDimensions(&input_dimensions[0],
                                      input_dimensions.size())
             : 1;
}

NNADAPTER_EXPORT void TransposeDimensions(
    int32_t* input_dimensions,
    const std::vector<int32_t>& permutation,
    int32_t* output_dimensions_ptr) {
  size_t permutation_count = permutation.size();
  std::vector<int32_t> origin_dimensions(input_dimensions,
                                         input_dimensions + permutation_count);
  for (size_t i = 0; i < permutation_count; i++) {
    auto dimension = origin_dimensions[permutation[i]];
    if (output_dimensions_ptr != nullptr) {
      output_dimensions_ptr[i] = dimension;
    } else {
      input_dimensions[i] = dimension;
    }
  }
}

NNADAPTER_EXPORT void ReshapeDimensions(int32_t* input_dimensions,
                                        uint32_t* input_dimension_count,
                                        const std::vector<int32_t>& dimensions,
                                        int32_t* output_dimensions_ptr,
                                        uint32_t* output_dimension_count_ptr) {
  const int64_t input_size =
      ProductionOfDimensions(input_dimensions, *input_dimension_count);
  bool all_positive = std::all_of(
      dimensions.cbegin(), dimensions.cend(), [](int32_t i) { return i > 0; });
  // Only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int32_t unk_dim_val = -1;
  const int32_t copy_dim_val = 0;
  std::vector<int32_t> output_dimensions(dimensions.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == unk_dim_val) {
      NNADAPTER_CHECK_EQ(unk_dim_idx, -1)
          << "Only one dimension value of 'dimensions' can be -1. But received "
             "new_dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] is also -1.";
      unk_dim_idx = i;
    } else if (dimensions[i] == copy_dim_val) {
      NNADAPTER_CHECK_LT(static_cast<uint32_t>(i), *input_dimension_count)
          << "The index of 0 in `dimensions` must be less than the "
             "input_dimension_count. But received dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] = 0, input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input_dimension_count = " << *input_dimension_count << ".";
    } else {
      NNADAPTER_CHECK_GT(dimensions[i], 0)
          << "Each dimension value of 'dimensions' must not be negative except "
             "one unknown dimension. But received dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] = " << dimensions[i] << ".";
    }
    capacity *= (dimensions[i] ? dimensions[i] : input_dimensions[i]);
    output_dimensions[i] = (dimensions[i] ? static_cast<int32_t>(dimensions[i])
                                          : input_dimensions[i]);
  }
  if (unk_dim_idx != -1) {
    if (all_positive) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dimensions = [-1, 8, 1, 1], dimensions = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_dimensions[0] = 0 the following
      // check will fail.
      output_dimensions[unk_dim_idx] = -input_size / capacity;
      NNADAPTER_CHECK_EQ(output_dimensions[unk_dim_idx] * capacity, -input_size)
          << "The 'dimensions' is invalid. The input size must be divisible by "
             "known capacity of 'dimensions'. But received input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input size = " << input_size << ", 'dimensions' is ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], known capacity of 'dimensions' is " << capacity << ".";
    } else {
      output_dimensions[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      NNADAPTER_CHECK_EQ(capacity, input_size)
          << "The 'dimensions' is invalid. The input size must be equal to the "
             "capacity of 'dimensions'. But received input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input size = " << input_size << ", 'dimensions' is ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], the capacity of 'dimensions' is " << capacity << ".";
    }
  }
  if (!output_dimensions_ptr || !output_dimension_count_ptr) {
    output_dimensions_ptr = input_dimensions;
    output_dimension_count_ptr = input_dimension_count;
  }
  *output_dimension_count_ptr = output_dimensions.size();
  memcpy(output_dimensions_ptr,
         &output_dimensions[0],
         output_dimensions.size() * sizeof(int32_t));
}

}  // namespace nnadapter
