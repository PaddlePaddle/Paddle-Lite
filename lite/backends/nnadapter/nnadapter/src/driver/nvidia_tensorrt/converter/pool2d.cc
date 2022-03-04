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

#include "operation/pool2d.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE);
  if (operation_type == NNADAPTER_MAX_POOL_2D) {
    NNADAPTER_CHECK(!flag) << "Not support return indices.";
  }
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  nvinfer1::PoolingType pool_type;
  if (operation_type == NNADAPTER_AVERAGE_POOL_2D) {
    pool_type = nvinfer1::PoolingType::kAVERAGE;
  } else {
    pool_type = nvinfer1::PoolingType::kMAX;
  }
  auto pool_layer = converter->network()->addPoolingNd(
      *input_tensor, pool_type, nvinfer1::Dims2(kernel_height, kernel_width));
  NNADAPTER_CHECK(pool_layer);
  if (operation_type == NNADAPTER_AVERAGE_POOL_2D) {
    pool_layer->setAverageCountExcludesPadding(!flag);
  }
  pool_layer->setPrePadding(nvinfer1::Dims2(pad_height_top, pad_width_left));
  pool_layer->setPostPadding(
      nvinfer1::Dims2(pad_height_bottom, pad_width_right));
  pool_layer->setStrideNd(nvinfer1::Dims2(stride_height, stride_width));
  auto output_tensor = pool_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
