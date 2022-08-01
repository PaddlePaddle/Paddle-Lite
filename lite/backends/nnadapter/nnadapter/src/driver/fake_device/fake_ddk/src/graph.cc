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

#include "fake_ddk/graph.h"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include "logging.h"  // NOLINT
#include "utility.h"  // NOLINT

namespace fake_ddk {

Graph::Graph() {}

Graph::Graph(const std::vector<uint8_t>& buffer) {
  FAKE_DDK_CHECK(deserialize_graph_from_buffer(this, buffer) ==
                 StatusType::SUCCESS);
}

Graph::~Graph() { Clear(); }

void Graph::Clear() {
  for (auto& tensor : tensors_) {
    if (tensor.lifetime != LifeTimeType::INPUT && tensor.buffer &&
        tensor.length > 0) {
      free(tensor.buffer);
      tensor.buffer = nullptr;
      tensor.length = 0;
    }
  }
  tensors_.clear();
  operators_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
}

Tensor* Graph::AddTensor(const TensorAttr& attr, void* data) {
  tensors_.emplace_back();
  auto tensor = &tensors_.back();
  tensor->attr = attr;
  if (data) {
    tensor->length = get_tensor_buffer_length(attr);
    tensor->buffer = malloc(tensor->length);
    FAKE_DDK_CHECK(tensor->buffer)
        << "Failed to allocate buffer for a constant tensor, out of memory!";
    memcpy(tensor->buffer, data, tensor->length);
    tensor->lifetime = LifeTimeType::CONSTANT;
  } else {
    tensor->length = 0;
    tensor->buffer = nullptr;
    tensor->lifetime = LifeTimeType::TEMPORARY_VARIABLE;
  }
  return tensor;
}

Operator* Graph::AddOperator(OperatorType type,
                             const std::vector<Tensor*>& input_tensors,
                             const std::vector<Tensor*>& output_tensors,
                             void* attr) {
  operators_.emplace_back();
  auto op = &operators_.back();
  auto input_count = input_tensors.size();
  auto output_count = output_tensors.size();
  switch (type) {
    case OperatorType::FAKE_DDK_CONV2D: {
      op->attr.conv2d_attr = *static_cast<Conv2DAttr*>(attr);
      FAKE_DDK_CHECK_EQ(input_count, 3);   // input, filter and bias
      FAKE_DDK_CHECK_EQ(output_count, 1);  // output
    } break;
    default:
      FAKE_DDK_LOG(FATAL) << "Unsupported op type " << static_cast<int>(type)
                          << "!";
      return nullptr;
  }
  op->type = type;
  op->input_tensors = input_tensors;
  op->output_tensors = output_tensors;
  return op;
}

int Graph::IdentifyInputsAndOutputs(
    const std::vector<Tensor*>& input_tensors,
    const std::vector<Tensor*>& output_tensors) {
  auto IsValid = [&](const Tensor* candidate) {
    bool found = false;
    for (auto& tensor : tensors_) {
      if (candidate == &tensor) {
        found = true;
        break;
      }
    }
    return found;
  };
  auto input_count = input_tensors.size();
  auto output_count = output_tensors.size();
  for (size_t i = 0; i < input_count; i++) {
    auto tensor = input_tensors[i];
    FAKE_DDK_CHECK(IsValid(tensor)) << "Failed to find input tensor " << i
                                    << " in graph !";
    FAKE_DDK_CHECK(tensor->lifetime != LifeTimeType::CONSTANT)
        << "Input tensor should not be a constant tensor!";
    tensor->lifetime = LifeTimeType::INPUT;
  }
  for (size_t i = 0; i < output_count; i++) {
    auto tensor = output_tensors[i];
    FAKE_DDK_CHECK(IsValid(tensor)) << "Failed to find output tensor " << i
                                    << " in graph !";
    FAKE_DDK_CHECK(tensor->lifetime != LifeTimeType::CONSTANT)
        << "Output tensor should not be a constant tensor!";
    tensor->lifetime = LifeTimeType::OUTPUT;
  }
  input_tensors_ = input_tensors;
  output_tensors_ = output_tensors;
  return StatusType::SUCCESS;
}

int Graph::QueryInputsAndOutputs(std::vector<TensorAttr>* input_attrs,
                                 std::vector<TensorAttr>* output_attrs) {
  auto input_count = input_tensors_.size();
  auto output_count = output_tensors_.size();
  input_attrs->resize(input_count);
  output_attrs->resize(output_count);
  for (size_t i = 0; i < input_count; i++) {
    input_attrs->at(i) = input_tensors_[i]->attr;
  }
  for (size_t i = 0; i < output_count; i++) {
    output_attrs->at(i) = output_tensors_[i]->attr;
  }
  return StatusType::SUCCESS;
}

}  // namespace fake_ddk
