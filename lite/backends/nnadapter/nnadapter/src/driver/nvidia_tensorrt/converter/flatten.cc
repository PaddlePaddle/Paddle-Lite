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

#include "operation/flatten.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertFlatten(Converter* converter, core::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (!IsOperandWithDynamicShape(input_operand)) {
    auto flatten_layer = converter->network()->addShuffle(*input_tensor);
    NNADAPTER_CHECK(flatten_layer);
    auto dims = ConvertToNVDims(output_operand->type.dimensions);
    flatten_layer->setReshapeDimensions(dims);
    converter->UpdateTensorMap(output_operand, flatten_layer->getOutput(0));
  } else {
    if (start_axis < 0) start_axis += input_operand->type.dimensions.count;
    if (end_axis < 0) end_axis += input_operand->type.dimensions.count;
    auto shape_layer = converter->network()->addShape(*input_tensor);
    auto shape_layer_itensor = shape_layer->getOutput(0);
    nvinfer1::Dims start_dim, size_dim, stride_dim;
    start_dim.nbDims = 1;
    size_dim.nbDims = 1;
    stride_dim.nbDims = 1;
    start_dim.d[0] = start_axis;
    size_dim.d[0] = end_axis - start_axis + 1;
    stride_dim.d[0] = 1;
    auto slice_layer = converter->network()->addSlice(
        *shape_layer_itensor, start_dim, size_dim, stride_dim);
    uint32_t reduce_dim = 1;
    auto reduce_prod_layer =
        converter->network()->addReduce(*(slice_layer->getOutput(0)),
                                        nvinfer1::ReduceOperation::kPROD,
                                        reduce_dim,
                                        true);
    nvinfer1::ITensor* output_shape_tensor = nullptr;
    if (start_axis == 0 &&
        end_axis == input_operand->type.dimensions.count - 1) {
      output_shape_tensor = reduce_prod_layer->getOutput(0);
    } else {
      std::vector<nvinfer1::ITensor*> itensors;
      if (start_axis > 0) {
        nvinfer1::Dims left_start_dim, left_size_dim, left_stride_dim;
        left_start_dim.nbDims = 1;
        left_size_dim.nbDims = 1;
        left_stride_dim.nbDims = 1;
        left_start_dim.d[0] = 0;
        left_size_dim.d[0] = start_axis;
        left_stride_dim.d[0] = 1;
        auto slice_layer_left =
            converter->network()->addSlice(*shape_layer_itensor,
                                           left_start_dim,
                                           left_size_dim,
                                           left_stride_dim);
        itensors.push_back(slice_layer_left->getOutput(0));
      }
      itensors.push_back(reduce_prod_layer->getOutput(0));
      if (end_axis < input_operand->type.dimensions.count - 1) {
        nvinfer1::Dims right_start_dim, right_size_dim, right_stride_dim;
        right_start_dim.nbDims = 1;
        right_size_dim.nbDims = 1;
        right_stride_dim.nbDims = 1;
        right_start_dim.d[0] = end_axis + 1;
        right_size_dim.d[0] =
            input_operand->type.dimensions.count - end_axis - 1;
        right_stride_dim.d[0] = 1;
        auto slice_layer_right =
            converter->network()->addSlice(*shape_layer_itensor,
                                           right_start_dim,
                                           right_size_dim,
                                           right_stride_dim);
        itensors.push_back(slice_layer_right->getOutput(0));
      }
      auto concat_layer = converter->network()->addConcatenation(
          itensors.data(), itensors.size());
      concat_layer->setAxis(0);
      output_shape_tensor = concat_layer->getOutput(0);
    }
    auto flatten_layer = converter->network()->addShuffle(*input_tensor);
    flatten_layer->setInput(1, *output_shape_tensor);
    auto output_tensor = flatten_layer->getOutput(0);
    converter->UpdateTensorMap(output_operand, output_tensor);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
