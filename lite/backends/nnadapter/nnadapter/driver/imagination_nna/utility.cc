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

#include "driver/imagination_nna/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace imagination_nna {

imgdnn_type ConvertPrecision(NNAdapterOperandPrecisionCode input_precision) {
  imgdnn_type output_precision;
  switch (input_precision) {
    case NNADAPTER_TENSOR_INT8:
      output_precision = IMGDNN_TYPE_I8;
      break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
      output_precision = IMGDNN_TYPE_Q_I8;
      break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = IMGDNN_TYPE_QPA_I8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = IMGDNN_TYPE_I16;
      break;
    case NNADAPTER_TENSOR_INT32:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = IMGDNN_TYPE_I32;
      break;
    case NNADAPTER_TENSOR_UINT8:
      output_precision = IMGDNN_TYPE_U8;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = IMGDNN_TYPE_Q_U8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = IMGDNN_TYPE_U16;
      break;
    case NNADAPTER_TENSOR_UINT32:
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER:
      output_precision = IMGDNN_TYPE_U32;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = IMGDNN_TYPE_F16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = IMGDNN_TYPE_F32;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to imgdnn_type !";
      break;
  }
  return output_precision;
}

imgdnn_dimensions_order ConvertDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  imgdnn_dimensions_order output_layout = IMGDNN_UNKNOWN;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = IMGDNN_NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = IMGDNN_NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to imgdnn_dimensions_order !";
      break;
  }
  return output_layout;
}

void ConvertDimensions(int32_t* input_dimensions,
                       uint32_t input_dimensions_count,
                       size_t* output_dimensions,
                       unsigned int* output_dimensions_count) {
  NNADAPTER_CHECK_LE(input_dimensions_count, IMGDNN_DESCRIPTOR_MAX_DIM);
  *output_dimensions_count = input_dimensions_count;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GT(dimension, 0);
    output_dimensions[i] = dimension;
  }
}

}  // namespace imagination_nna
}  // namespace nnadapter
