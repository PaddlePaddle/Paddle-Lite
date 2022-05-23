// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/batch_normalization.h"
#include <cmath>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  float* scale_ptr = reinterpret_cast<float*>(scale_operand->buffer);
  float* bias_ptr = reinterpret_cast<float*>(bias_operand->buffer);
  float* mean_ptr = reinterpret_cast<float*>(mean_operand->buffer);
  float* var_ptr = reinterpret_cast<float*>(variance_operand->buffer);
  NNADAPTER_CHECK(scale_ptr);
  NNADAPTER_CHECK(bias_ptr);
  NNADAPTER_CHECK(mean_ptr);
  NNADAPTER_CHECK(var_ptr);
  auto input_tensor_dim = input_tensor->getDimensions();
  // Add shuffle operator to reshape data into 3 dimensions
  if (input_tensor_dim.nbDims < 3) {
    nvinfer1::Dims unsqueeze_shape;
    unsqueeze_shape.nbDims = 3;
    for (int i = 0; i < 3; i++) {
      if (i < input_tensor_dim.nbDims) {
        unsqueeze_shape.d[i] =
            input_tensor_dim.d[i] < 0 ? 0 : input_tensor_dim.d[i];
      } else {
        unsqueeze_shape.d[i] = 1;
      }
    }
    auto unsqueeze_layer = converter->network()->addShuffle(*input_tensor);
    unsqueeze_layer->setReshapeDimensions(unsqueeze_shape);
    input_tensor = unsqueeze_layer->getOutput(0);
  }
  // Add batch_normalization op using ScaleNd operator
  NNADAPTER_CHECK_EQ(scale_operand->type.dimensions.data[0],
                     input_tensor_dim.d[0]);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.data[0],
                     input_tensor_dim.d[0]);
  NNADAPTER_CHECK_EQ(mean_operand->type.dimensions.data[0],
                     input_tensor_dim.d[0]);
  NNADAPTER_CHECK_EQ(variance_operand->type.dimensions.data[0],
                     input_tensor_dim.d[0]);
  std::vector<float> fuse_scale(input_tensor_dim.d[0], 0);
  std::vector<float> fuse_bias(input_tensor_dim.d[0], 0);
  auto fuse_scale_ptr = fuse_scale.data();
  auto fuse_bias_ptr = fuse_bias.data();
  for (int i = 0; i < input_tensor_dim.d[0]; i++) {
    fuse_scale_ptr[i] = scale_ptr[i] / sqrtf(var_ptr[i] + epsilon);
    fuse_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * fuse_scale_ptr[i];
  }
  const float* fuse_scale_ptr_const = fuse_scale_ptr;
  const float* fuse_bias_ptr_const = fuse_bias_ptr;
  const float* power_ptr = nullptr;
  // add scale op
  nvinfer1::Weights scale_w =
      converter->AddWeights(fuse_scale_ptr_const, input_tensor_dim.d[0]);
  nvinfer1::Weights shift_w =
      converter->AddWeights(fuse_bias_ptr_const, input_tensor_dim.d[0]);
  nvinfer1::Weights power_w = converter->AddWeights(power_ptr, 0);
  auto layer = converter->network()->addScaleNd(*input_tensor,
                                                nvinfer1::ScaleMode::kCHANNEL,
                                                shift_w,
                                                scale_w,
                                                power_w,
                                                0);
  auto output_tensor = layer->getOutput(0);
  // Add shuffle operator to recover shape
  if (input_tensor_dim.nbDims < 3) {
    nvinfer1::Dims squeeze_shape;
    squeeze_shape.nbDims = input_tensor_dim.nbDims;
    for (int i = 0; i < squeeze_shape.nbDims; i++) {
      squeeze_shape.d[i] =
          input_tensor_dim.d[i] < 0 ? 0 : input_tensor_dim.d[i];
    }
    auto squeeze_layer =
        converter->network()->addShuffle(*(layer->getOutput(0)));
    squeeze_layer->setReshapeDimensions(squeeze_shape);
    output_tensor = squeeze_layer->getOutput(0);
  }
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
