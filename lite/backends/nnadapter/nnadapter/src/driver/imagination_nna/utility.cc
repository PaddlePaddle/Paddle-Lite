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

imgdnn_type ConvertToImgdnnPrecision(
    NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_INT8:
      return IMGDNN_TYPE_I8;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
      return IMGDNN_TYPE_Q_I8;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return IMGDNN_TYPE_QPA_I8;
    case NNADAPTER_INT16:
      return IMGDNN_TYPE_I16;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      return IMGDNN_TYPE_I32;
    case NNADAPTER_UINT8:
      return IMGDNN_TYPE_U8;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return IMGDNN_TYPE_Q_U8;
    case NNADAPTER_UINT16:
      return IMGDNN_TYPE_U16;
    case NNADAPTER_UINT32:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      return IMGDNN_TYPE_U32;
    case NNADAPTER_FLOAT16:
      return IMGDNN_TYPE_F16;
    case NNADAPTER_FLOAT32:
      return IMGDNN_TYPE_F32;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to imgdnn_type !";
      break;
  }
  return IMGDNN_TYPE_F32;
}

imgdnn_dimensions_order ConvertToImgdnnDataLayout(
    NNAdapterOperandLayoutCode layout_code) {
  switch (layout_code) {
    case NNADAPTER_NCHW:
      return IMGDNN_NCHW;
    case NNADAPTER_NHWC:
      return IMGDNN_NHWC;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(layout_code)
          << ") to imgdnn_dimensions_order !";
      break;
  }
  return IMGDNN_UNKNOWN;
}

void ConvertToImgdnnDimensions(int32_t* input_dimensions,
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
