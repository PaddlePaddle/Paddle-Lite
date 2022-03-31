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

#pragma once

#include <vector>
#include "fake_ddk/graph.h"

namespace fake_ddk {

/* Get the slice of the shape */
std::vector<int32_t> shape_slice(const std::vector<int32_t>& input_shape,
                                 int start,
                                 int end);

/* Get the production of the shape */
int64_t shape_production(const std::vector<int32_t>& input_shape);

/* Get the number of bytes of the precision type */
int64_t get_tensor_precision_bytes(PrecisionType precision);

/* Get the buffer size of the tensor */
int64_t get_tensor_buffer_length(const TensorAttr& attr);

/* Get or allocate buffer based on the precision type and shape of the tensor */
void* get_tensor_buffer_address(Tensor* tensor);

/* symmetric per-layer int8 quantization */
int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int8_t* output_data);
/* symmetric per-channel int8 quantization */
int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             const std::vector<float>& output_scales,
             int32_t output_channel_dim,
             int8_t* output_data);
/* asymmetric per-layer uint8 quantization */
int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int32_t output_zero_point,
             uint8_t* output_data);
/* symmetric per-layer int32 quantization */
int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int32_t* output_data);
/* symmetric per-channel int32 quantization */
int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             const std::vector<float>& output_scales,
             int32_t output_channel_dim,
             int32_t* output_data);

/* symmetric per-layer int8 dequantization */
int dequantize(int8_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               float* output_data);
/* symmetric per-channel int8 dequantization */
int dequantize(int8_t* input_data,
               const std::vector<int32_t>& input_shape,
               const std::vector<float>& input_scales,
               int32_t input_channel_dim,
               float* output_data);
/* asymmetric per-layer uint8 dequantization */
int dequantize(uint8_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               int32_t input_zero_point,
               float* output_data);
/* symmetric per-layer int32 dequantization */
int dequantize(int32_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               float* output_data);
/* symmetric per-channel int32 dequantization */
int dequantize(int32_t* input_data,
               const std::vector<int32_t>& input_shape,
               const std::vector<float>& input_scales,
               int input_channel_dim,
               float* output_data);

/* Sort the operators of the graph in topological order.
  * @return the sorted operators.
*/
std::vector<Operator*> sort_operators_in_topological_order(Graph* graph);

/* Serialize graph to buffer.
  * @return StatusType::SUCCESS when success.
*/
int serialize_graph_to_buffer(const Graph& graph, std::vector<uint8_t>* buffer);

/* Derialize graph from buffer.
  * @return StatusType::SUCCESS when success.
*/
int deserialize_graph_from_buffer(Graph* graph,
                                  const std::vector<uint8_t>& buffer);

}  // namespace fake_ddk
