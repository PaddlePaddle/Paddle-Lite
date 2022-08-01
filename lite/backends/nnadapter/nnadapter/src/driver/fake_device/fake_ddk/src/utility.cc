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

#include "utility.h"  // NOLINT
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include "logging.h"  // NOLINT
#include "utility.h"  // NOLINT

namespace fake_ddk {

std::vector<int32_t> shape_slice(const std::vector<int32_t>& input_shape,
                                 int start,
                                 int end) {
  int input_rank = input_shape.size();
  start = start < 0 ? 0 : (start > input_rank ? input_rank : start);
  end = end < start ? start : (end > input_rank ? input_rank : end);
  return std::vector<int32_t>(input_shape.data() + start,
                              input_shape.data() + end);
}

int64_t shape_production(const std::vector<int32_t>& input_shape) {
  auto input_rank = input_shape.size();
  int64_t production = 1;
  for (size_t i = 0; i < input_rank; i++) {
    auto dimension = input_shape[i];
    production *= dimension;
  }
  return production;
}

int64_t get_tensor_precision_bytes(PrecisionType precision) {
  switch (precision) {
    case PrecisionType::BOOL8:
    case PrecisionType::INT8:
    case PrecisionType::UINT8:
    case PrecisionType::QUANT_INT8_SYMM_PER_LAYER:
    case PrecisionType::QUANT_INT8_SYMM_PER_CHANNEL:
    case PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER:
      return 1;
    case PrecisionType::INT16:
    case PrecisionType::UINT16:
    case PrecisionType::FLOAT16:
      return 2;
    case PrecisionType::INT32:
    case PrecisionType::UINT32:
    case PrecisionType::FLOAT32:
    case PrecisionType::QUANT_INT32_SYMM_PER_LAYER:
    case PrecisionType::QUANT_INT32_SYMM_PER_CHANNEL:
      return 4;
    case PrecisionType::INT64:
    case PrecisionType::UINT64:
    case PrecisionType::FLOAT64:
      return 8;
    default:
      FAKE_DDK_LOG(FATAL) << "Unsupported precision type "
                          << static_cast<int>(precision);
      break;
  }
  return 0;
}

int64_t get_tensor_buffer_length(const TensorAttr& attr) {
  int64_t production = shape_production(attr.shape);
  return get_tensor_precision_bytes(attr.precision) * production;
}

void* get_tensor_buffer_address(Tensor* tensor) {
  auto length = get_tensor_buffer_length(tensor->attr);
  if (tensor->length < length) {
    FAKE_DDK_CHECK(tensor->lifetime == LifeTimeType::TEMPORARY_VARIABLE ||
                   tensor->lifetime == LifeTimeType::OUTPUT);
    if (tensor->buffer) {
      free(tensor->buffer);
    }
    tensor->buffer = malloc(length);
    FAKE_DDK_CHECK(tensor->buffer) << "Failed to allocate " << length
                                   << " bytes, out of memory!";
    tensor->length = length;
  }
  FAKE_DDK_CHECK(tensor->buffer);
  return tensor->buffer;
}

template <typename T>
static int quantize(float* input_data,
                    const std::vector<int32_t>& input_shape,
                    const std::vector<float>& output_scales,
                    const std::vector<int32_t>& output_zero_points,
                    int32_t output_channel_dim,
                    int dtype_min,
                    int dtype_max,
                    T* output_data) {
  if (!input_data || input_shape.empty() || output_scales.empty() ||
      output_zero_points.empty() || !output_data) {
    return StatusType::FAILURE;
  }
  auto input_rank = input_shape.size();
  auto input_count = shape_production(input_shape);
  auto scale_count = output_scales.size();
  if (scale_count != output_zero_points.size()) {
    return StatusType::FAILURE;
  }
  auto channel_dim = output_channel_dim;
  if (scale_count > 1 && channel_dim < 0) {
    return StatusType::FAILURE;
  }
  int64_t outer_count = input_count;
  int64_t inner_count = 1;
  if (scale_count > 1 && channel_dim >= 0) {
    auto channel_count = input_shape[channel_dim];
    if (channel_count != scale_count) {
      return StatusType::FAILURE;
    }
    outer_count = shape_production(shape_slice(input_shape, 0, channel_dim));
    inner_count =
        shape_production(shape_slice(input_shape, channel_dim + 1, input_rank));
  }
  for (int64_t i = 0; i < outer_count; i++) {
    for (size_t j = 0; j < scale_count; j++) {
      for (int64_t k = 0; k < inner_count; k++) {
        auto index = i * scale_count * inner_count + j * inner_count + k;
        output_data[index] = std::min(
            std::max(static_cast<int>(input_data[index] / output_scales[j]) +
                         output_zero_points[j],
                     dtype_min),
            dtype_max);
      }
    }
  }
  return StatusType::SUCCESS;
}

int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int8_t* output_data) {
  return quantize<int8_t>(input_data,
                          input_shape,
                          std::vector<float>({output_scale}),
                          std::vector<int32_t>({0}),
                          -1,
                          -127,
                          127,
                          output_data);
}

int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             const std::vector<float>& output_scales,
             int32_t output_channel_dim,
             int8_t* output_data) {
  return quantize<int8_t>(input_data,
                          input_shape,
                          output_scales,
                          std::vector<int32_t>(output_scales.size(), 0),
                          output_channel_dim,
                          -127,
                          127,
                          output_data);
}

int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int32_t output_zero_point,
             uint8_t* output_data) {
  return quantize<uint8_t>(input_data,
                           input_shape,
                           std::vector<float>({output_scale}),
                           std::vector<int32_t>({output_zero_point}),
                           -1,
                           0,
                           255,
                           output_data);
}

int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             float output_scale,
             int32_t* output_data) {
  return quantize<int32_t>(input_data,
                           input_shape,
                           std::vector<float>({output_scale}),
                           std::vector<int32_t>({0}),
                           -1,
                           -2147483647,
                           2147483647,
                           output_data);
}

int quantize(float* input_data,
             const std::vector<int32_t>& input_shape,
             const std::vector<float>& output_scales,
             int32_t output_channel_dim,
             int32_t* output_data) {
  return quantize<int32_t>(input_data,
                           input_shape,
                           output_scales,
                           std::vector<int32_t>(output_scales.size(), 0),
                           output_channel_dim,
                           -2147483647,
                           2147483647,
                           output_data);
}

template <typename T>
static int dequantize(T* input_data,
                      const std::vector<int32_t>& input_shape,
                      const std::vector<float>& input_scales,
                      const std::vector<int32_t>& input_zero_points,
                      int32_t input_channel_dim,
                      int dtype_min,
                      int dtype_max,
                      float* output_data) {
  if (!input_data || input_shape.empty() || input_scales.empty() ||
      input_zero_points.empty() || !output_data) {
    return StatusType::FAILURE;
  }
  auto input_rank = input_shape.size();
  auto input_count = shape_production(input_shape);
  auto scale_count = input_scales.size();
  if (scale_count != input_zero_points.size()) {
    return StatusType::FAILURE;
  }
  auto channel_dim = input_channel_dim;
  if (scale_count > 1 && channel_dim < 0) {
    return StatusType::FAILURE;
  }
  int64_t outer_count = input_count;
  int64_t inner_count = 1;
  if (scale_count > 1 && channel_dim >= 0) {
    auto channel_count = input_shape[channel_dim];
    if (channel_count != scale_count) {
      return StatusType::FAILURE;
    }
    outer_count = shape_production(shape_slice(input_shape, 0, channel_dim));
    inner_count =
        shape_production(shape_slice(input_shape, channel_dim + 1, input_rank));
  }
  for (int64_t i = 0; i < outer_count; i++) {
    for (size_t j = 0; j < scale_count; j++) {
      for (int64_t k = 0; k < inner_count; k++) {
        auto index = i * scale_count * inner_count + j * inner_count + k;
        output_data[index] =
            (static_cast<float>(std::min(
                 std::max(static_cast<int>(input_data[index]), dtype_min),
                 dtype_max)) -
             input_zero_points[j]) *
            input_scales[j];
      }
    }
  }
  return StatusType::SUCCESS;
}

int dequantize(int8_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               float* output_data) {
  return dequantize<int8_t>(input_data,
                            input_shape,
                            std::vector<float>({input_scale}),
                            std::vector<int32_t>({0}),
                            -1,
                            -127,
                            127,
                            output_data);
}

int dequantize(int8_t* input_data,
               const std::vector<int32_t>& input_shape,
               const std::vector<float>& input_scales,
               int32_t input_channel_dim,
               float* output_data) {
  return dequantize<int8_t>(input_data,
                            input_shape,
                            input_scales,
                            std::vector<int32_t>(input_scales.size(), 0),
                            input_channel_dim,
                            -127,
                            127,
                            output_data);
}

int dequantize(uint8_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               int32_t input_zero_point,
               float* output_data) {
  return dequantize<uint8_t>(input_data,
                             input_shape,
                             std::vector<float>({input_scale}),
                             std::vector<int32_t>({input_zero_point}),
                             -1,
                             0,
                             255,
                             output_data);
}

int dequantize(int32_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               float* output_data) {
  return dequantize<int32_t>(input_data,
                             input_shape,
                             std::vector<float>({input_scale}),
                             std::vector<int32_t>({0}),
                             -1,
                             -2147483647,
                             2147483647,
                             output_data);
}

int dequantize(int32_t* input_data,
               const std::vector<int32_t>& input_shape,
               const std::vector<float>& input_scales,
               int input_channel_dim,
               float* output_data) {
  return dequantize<int32_t>(input_data,
                             input_shape,
                             input_scales,
                             std::vector<int32_t>(input_scales.size(), 0),
                             input_channel_dim,
                             -2147483647,
                             2147483647,
                             output_data);
}

std::vector<Operator*> sort_operators_in_topological_order(Graph* graph) {
  FAKE_DDK_VLOG(5) << "graph total tensors: " << graph->tensors_.size();
  FAKE_DDK_VLOG(5) << "graph input tensors: " << graph->input_tensors_.size();
  FAKE_DDK_VLOG(5) << "graph output tensors: " << graph->output_tensors_.size();
  FAKE_DDK_VLOG(5) << "graph total operators: " << graph->operators_.size();
  /* Operators in topological order */
  std::vector<Operator*> operators;
  std::vector<Operator*> queue;
  /* Use to find all of adjacent operators according to a given tensor. */
  std::multimap<Tensor*, Operator*> map;
  /* The counters of variable inputs for all of operations. */
  std::map<Operator*, uint32_t> counts;
  for (auto& op : graph->operators_) {
    uint32_t count = 0;
    for (auto tensor : op.input_tensors) {
      LifeTimeType lifetime{LifeTimeType::CONSTANT};
      if (tensor != nullptr) {
        lifetime = tensor->lifetime;
      }
      if (lifetime == LifeTimeType::TEMPORARY_VARIABLE ||
          lifetime == LifeTimeType::OUTPUT) {
        count++;
        map.insert(std::pair<Tensor*, Operator*>(tensor, &op));
      }
    }
    if (count == 0) {
      /* The operator which only depends the model inputs and constants */
      queue.push_back(&op);
    }
    counts[&op] = count;
  }
  while (queue.size() > 0) {
    auto op = queue.back();
    queue.pop_back();
    operators.push_back(op);
    for (auto tensor : op->output_tensors) {
      auto range = map.equal_range(tensor);
      for (auto i = range.first; i != range.second; i++) {
        uint32_t& count = counts[i->second];
        if (--count == 0) {
          queue.push_back(i->second);
        }
      }
    }
  }
  return operators;
}

inline void serialize_data(std::vector<uint8_t>* buffer,
                           size_t* offset,
                           const void* data,
                           size_t size) {
  memcpy(buffer->data() + *offset, data, size);
  *offset += size;
}

template <typename T>
void serialize_data(std::vector<uint8_t>* buffer, size_t* offset, T value) {
  serialize_data(buffer, offset, &value, sizeof(T));
}

template <typename T>
void serialize_data(std::vector<uint8_t>* buffer,
                    size_t* offset,
                    const std::vector<T>& values) {
  size_t count = values.size();
  serialize_data(buffer, offset, count);
  if (count > 0) {
    serialize_data(buffer, offset, values.data(), sizeof(T) * count);
  }
}

inline void deserialize_data(const std::vector<uint8_t>& buffer,
                             size_t* offset,
                             void* data,
                             size_t size) {
  memcpy(data, buffer.data() + *offset, size);
  *offset += size;
}

template <typename T>
void deserialize_data(const std::vector<uint8_t>& buffer,
                      size_t* offset,
                      T* value) {
  deserialize_data(buffer, offset, value, sizeof(T));
}

template <typename T>
void deserialize_data(const std::vector<uint8_t>& buffer,
                      size_t* offset,
                      std::vector<T>* values) {
  size_t count = 0;
  deserialize_data(buffer, offset, &count);
  values->clear();
  if (count > 0) {
    values->resize(count);
    deserialize_data(buffer, offset, values->data(), sizeof(T) * count);
  }
}

/* Get the buffer size used to serialize the graph */
inline size_t get_serialized_graph_buffer_length(const Graph& graph) {
  // Serialize the tensors
  size_t length = 0;
  length += sizeof(size_t);
  for (auto& tensor : graph.tensors_) {
    length += sizeof(PrecisionType);
    length += sizeof(DataLayoutType);
    length += sizeof(size_t);
    length += sizeof(int32_t) * tensor.attr.shape.size();
    length += sizeof(size_t);
    length += sizeof(float) * tensor.attr.quant_params.scales.size();
    length += sizeof(size_t);
    length += sizeof(int32_t) * tensor.attr.quant_params.zero_points.size();
    length += sizeof(int32_t);
    length += sizeof(LifeTimeType);
    if (tensor.lifetime == LifeTimeType::CONSTANT) {
      length += sizeof(size_t);
      length += tensor.length;
    }
  }
  // Serialize the operators
  length += sizeof(size_t);
  for (auto& op : graph.operators_) {
    length += sizeof(OperatorType);
    length += sizeof(OperatorAttr);
    length += sizeof(size_t);
    length += sizeof(int64_t) * op.input_tensors.size();
    length += sizeof(size_t);
    length += sizeof(int64_t) * op.output_tensors.size();
  }
  // Serialize the indexes of the input tensors of the graph
  length += sizeof(size_t);
  length += sizeof(int64_t) * graph.input_tensors_.size();
  // Serialize the indexes of the output tensors of the graph
  length += sizeof(size_t);
  length += sizeof(int64_t) * graph.output_tensors_.size();
  return length;
}

int serialize_graph_to_buffer(const Graph& graph,
                              std::vector<uint8_t>* buffer) {
  if (!buffer) {
    return StatusType::FAILURE;
  }
  // Get the buffer size used to serialize the graph and prepare the buffer
  auto length = get_serialized_graph_buffer_length(graph);
  FAKE_DDK_CHECK_GT(length, 0);
  buffer->resize(length);
  // Serialize the tensors
  size_t offset = 0;
  serialize_data(buffer, &offset, graph.tensors_.size());
  int64_t tensor_index = 0;
  std::unordered_map<const Tensor*, int64_t> tensor_to_index;
  tensor_to_index[0] = -1;  // Map to -1 if tensor is nullptr
  for (auto& tensor : graph.tensors_) {
    serialize_data(buffer, &offset, tensor.attr.precision);
    serialize_data(buffer, &offset, tensor.attr.layout);
    serialize_data(buffer, &offset, tensor.attr.shape);
    serialize_data(buffer, &offset, tensor.attr.quant_params.scales);
    serialize_data(buffer, &offset, tensor.attr.quant_params.zero_points);
    serialize_data(buffer, &offset, tensor.attr.quant_params.channel_dim);
    serialize_data(buffer, &offset, tensor.lifetime);
    if (tensor.lifetime == LifeTimeType::CONSTANT) {
      FAKE_DDK_CHECK(tensor.buffer);
      serialize_data(buffer, &offset, tensor.length);
      serialize_data(buffer, &offset, tensor.buffer, tensor.length);
    }
    tensor_to_index[&tensor] = tensor_index++;
  }
  // Serialize the operators
  serialize_data(buffer, &offset, graph.operators_.size());
  for (auto& op : graph.operators_) {
    serialize_data(buffer, &offset, op.type);
    serialize_data(buffer, &offset, &op.attr, sizeof(OperatorAttr));
    // Serialize the indexes of the input tensors of the operator
    serialize_data(buffer, &offset, op.input_tensors.size());
    for (auto tensor : op.input_tensors) {
      FAKE_DDK_CHECK(tensor_to_index.count(tensor)) << "Tensor @" << std::hex
                                                    << tensor << " not found!";
      serialize_data(buffer, &offset, tensor_to_index[tensor]);
    }
    // Serialize the indexes of the output tensors of the operator
    serialize_data(buffer, &offset, op.output_tensors.size());
    for (auto tensor : op.output_tensors) {
      FAKE_DDK_CHECK(tensor_to_index.count(tensor)) << "Tensor @" << std::hex
                                                    << tensor << " not found!";
      serialize_data(buffer, &offset, tensor_to_index[tensor]);
    }
  }
  // Serialize the indexes of the input tensors of the graph
  serialize_data(buffer, &offset, graph.input_tensors_.size());
  for (auto tensor : graph.input_tensors_) {
    FAKE_DDK_CHECK(tensor_to_index.count(tensor)) << "Tensor @" << std::hex
                                                  << tensor << " not found!";
    serialize_data(buffer, &offset, tensor_to_index[tensor]);
  }
  // Serialize the indexes of the output tensors of the graph
  serialize_data(buffer, &offset, graph.output_tensors_.size());
  for (auto tensor : graph.output_tensors_) {
    FAKE_DDK_CHECK(tensor_to_index.count(tensor)) << "Tensor @" << std::hex
                                                  << tensor << " not found!";
    serialize_data(buffer, &offset, tensor_to_index[tensor]);
  }
  FAKE_DDK_CHECK_EQ(offset, length);
  return StatusType::SUCCESS;
}

int deserialize_graph_from_buffer(Graph* graph,
                                  const std::vector<uint8_t>& buffer) {
  if (!graph || buffer.size() == 0) {
    return StatusType::FAILURE;
  }
  graph->Clear();
  // Deserialize the tensors
  size_t offset = 0;
  size_t tensor_count = 0;
  deserialize_data(buffer, &offset, &tensor_count);
  std::vector<Tensor*> index_to_tensor(tensor_count);
  for (size_t i = 0; i < tensor_count; i++) {
    graph->tensors_.emplace_back();
    auto tensor = &graph->tensors_.back();
    deserialize_data(buffer, &offset, &tensor->attr.precision);
    deserialize_data(buffer, &offset, &tensor->attr.layout);
    deserialize_data(buffer, &offset, &tensor->attr.shape);
    deserialize_data(buffer, &offset, &tensor->attr.quant_params.scales);
    deserialize_data(buffer, &offset, &tensor->attr.quant_params.zero_points);
    deserialize_data(buffer, &offset, &tensor->attr.quant_params.channel_dim);
    deserialize_data(buffer, &offset, &tensor->lifetime);
    if (tensor->lifetime == LifeTimeType::CONSTANT) {
      deserialize_data(buffer, &offset, &tensor->length);
      tensor->buffer = reinterpret_cast<void*>(malloc(tensor->length));
      FAKE_DDK_CHECK(tensor->buffer)
          << "Failed to allocate the buffer for a constant tensor!";
      deserialize_data(buffer, &offset, tensor->buffer, tensor->length);
    }
    index_to_tensor[i] = tensor;
  }
  // Deserialize the operators
  size_t operator_count = 0;
  deserialize_data(buffer, &offset, &operator_count);
  for (size_t i = 0; i < operator_count; i++) {
    graph->operators_.emplace_back();
    auto op = &graph->operators_.back();
    deserialize_data(buffer, &offset, &op->type);
    deserialize_data(buffer, &offset, &op->attr, sizeof(OperatorAttr));
    // Deserialize the indexes of the input tensors of the operator
    size_t input_tensor_count = 0;
    deserialize_data(buffer, &offset, &input_tensor_count);
    op->input_tensors.resize(input_tensor_count);
    for (size_t j = 0; j < input_tensor_count; j++) {
      int64_t tensor_index;
      deserialize_data(buffer, &offset, &tensor_index);
      FAKE_DDK_CHECK(tensor_index == -1 ||
                     (tensor_index >= 0 && tensor_index < tensor_count));
      op->input_tensors[j] =
          tensor_index == -1 ? nullptr : index_to_tensor[tensor_index];
    }
    // Deserialize the indexes of the output tensors of the operator
    size_t output_tensor_count = 0;
    deserialize_data(buffer, &offset, &output_tensor_count);
    op->output_tensors.resize(output_tensor_count);
    for (size_t j = 0; j < output_tensor_count; j++) {
      int64_t tensor_index;
      deserialize_data(buffer, &offset, &tensor_index);
      FAKE_DDK_CHECK(tensor_index == -1 ||
                     (tensor_index >= 0 && tensor_index < tensor_count));
      op->output_tensors[j] =
          tensor_index == -1 ? nullptr : index_to_tensor[tensor_index];
    }
  }
  // Deserialize the indexes of the input tensors of the graph
  size_t input_tensor_count = 0;
  deserialize_data(buffer, &offset, &input_tensor_count);
  graph->input_tensors_.resize(input_tensor_count);
  for (size_t i = 0; i < input_tensor_count; i++) {
    int64_t tensor_index;
    deserialize_data(buffer, &offset, &tensor_index);
    FAKE_DDK_CHECK(tensor_index == -1 ||
                   (tensor_index >= 0 && tensor_index < tensor_count));
    graph->input_tensors_[i] =
        tensor_index == -1 ? nullptr : index_to_tensor[tensor_index];
  }
  // Deserialize the indexes of the output tensors of the graph
  size_t output_tensor_count = 0;
  deserialize_data(buffer, &offset, &output_tensor_count);
  graph->output_tensors_.resize(output_tensor_count);
  for (size_t i = 0; i < output_tensor_count; i++) {
    int64_t tensor_index;
    deserialize_data(buffer, &offset, &tensor_index);
    FAKE_DDK_CHECK(tensor_index == -1 ||
                   (tensor_index >= 0 && tensor_index < tensor_count));
    graph->output_tensors_[i] =
        tensor_index == -1 ? nullptr : index_to_tensor[tensor_index];
  }
  FAKE_DDK_CHECK_EQ(offset, buffer.size());
  return StatusType::SUCCESS;
}

}  // namespace fake_ddk
