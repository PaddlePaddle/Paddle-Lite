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

#include "driver/nvidia_tensorrt/converter/plugin/prior_box.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/prior_box.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertPriorBox(Converter* converter, core::Operation* operation) {
  PRIOR_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto image_tensor = converter->GetMappedTensor(image_operand);
  if (!image_tensor) {
    image_tensor = converter->ConvertOperand(image_operand);
  }
  auto input_dimension = input_operand->type.dimensions.data;
  auto image_dimension = image_operand->type.dimensions.data;
  std::vector<int32_t> input_dimension_vec(input_dimension,
                                           input_dimension + 4);
  std::vector<int32_t> image_dimension_vec(image_dimension,
                                           image_dimension + 4);
  PriorBoxPluginDynamic prior_box_plugin(
      aspect_ratios,
      input_dimension_vec,
      image_dimension_vec,
      *reinterpret_cast<float*>(step_w_operand->buffer),
      *reinterpret_cast<float*>(step_h_operand->buffer),
      min_sizes,
      max_sizes,
      *reinterpret_cast<float*>(offset_operand->buffer),
      *reinterpret_cast<bool*>(clip_operand->buffer),
      *reinterpret_cast<bool*>(flip_operand->buffer),
      *reinterpret_cast<bool*>(min_max_aspect_ratios_order_operand->buffer),
      variances);
  std::vector<nvinfer1::ITensor*> tensors{input_tensor, image_tensor};
  auto prior_box_layer =
      converter->network()->addPluginV2(tensors.data(), 2, prior_box_plugin);
  NNADAPTER_CHECK(prior_box_layer);
  auto boxes_tensor = prior_box_layer->getOutput(0);
  converter->UpdateTensorMap(boxes_operand, boxes_tensor);
  auto Variances_tensor = prior_box_layer->getOutput(1);
  converter->UpdateTensorMap(Variances_operand, Variances_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
