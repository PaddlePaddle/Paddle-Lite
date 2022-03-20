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

#include "driver/google_xnnpack/utility.h"
#include <limits>
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace google_xnnpack {

int XNNTensorDataTypeLength(xnn_datatype data_type) {
  switch (data_type) {
    case xnn_datatype_fp32:
      return 4;
    case xnn_datatype_fp16:
      return 2;
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qcint8:
      return 1;
    case xnn_datatype_qint32:
    case xnn_datatype_qcint32:
      return 4;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to get the length of XNNPACK data type("
                           << static_cast<int>(data_type) << ")!";
      break;
  }
  return 0;
}

xnn_datatype ConvertToXNNDataType(
    NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_FLOAT32:
      return xnn_datatype_fp32;
    case NNADAPTER_FLOAT16:
      return xnn_datatype_fp16;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
      return xnn_datatype_qint8;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return xnn_datatype_qcint8;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return xnn_datatype_quint8;
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      return xnn_datatype_qint32;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to the XNNPACK datatype!";
      break;
  }
  return xnn_datatype_invalid;
}

void ConvertToXNNDimensions(int32_t* input_dimensions,
                            uint32_t input_dimensions_count,
                            size_t* output_dimensions,
                            size_t* output_dimensions_count) {
  *output_dimensions_count = input_dimensions_count;
  for (size_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GE(dimension, 0);
    output_dimensions[i] = static_cast<size_t>(dimension);
  }
}

std::vector<size_t> ConvertToXNNDimensions(int32_t* input_dimensions,
                                           uint32_t input_dimensions_count) {
  std::vector<size_t> output_dimensions(input_dimensions_count);
  for (size_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GE(dimension, 0);
    output_dimensions[i] = static_cast<size_t>(dimension);
  }
  return output_dimensions;
}

bool ConvertFuseCodeToXNNClippingRange(int32_t fuse_code,
                                       float* clipping_min,
                                       float* clipping_max) {
  switch (fuse_code) {
    case NNADAPTER_FUSED_NONE:
      *clipping_min = -std::numeric_limits<float>::infinity();
      *clipping_max = std::numeric_limits<float>::infinity();
      return true;
    case NNADAPTER_FUSED_RELU:
      *clipping_min = 0.0f;
      *clipping_max = std::numeric_limits<float>::infinity();
      return true;
    case NNADAPTER_FUSED_RELU1:
      *clipping_min = 0.0f;
      *clipping_max = 1.0f;
      return true;
    case NNADAPTER_FUSED_RELU6:
      *clipping_min = 0.0f;
      *clipping_max = 6.0f;
      return true;
    default:
      NNADAPTER_LOG(FATAL) << "Unhandled case: fuse_code=" << fuse_code;
      break;
  }
  return false;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
