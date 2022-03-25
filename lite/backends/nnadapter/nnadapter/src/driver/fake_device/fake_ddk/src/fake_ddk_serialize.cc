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

#include <stdlib.h>
#include <string.h>
#include "fake_ddk/fake_ddk_pub.h"

namespace fake_ddk {
namespace nn {

int CalculateModelbufferSize(fakedevice_nn_graph_t *graph) {
  int buffer_size = 0;
  int node_num = graph->node_table.size();

  buffer_size += sizeof(uint32_t);  // node num
  for (int i = 0; i < node_num; i++) {
    fakedevice_nn_node_t *node = graph->node_table[i];
    buffer_size += sizeof(fakedevice_nn_op_t);     // op type
    buffer_size += sizeof(fakedevice_nn_param_t);  // op param
    buffer_size += sizeof(uint32_t);               // node's input-tensor num
    for (int j = 0; j < node->input_tensors.size(); j++) {
      buffer_size += sizeof(uint32_t);  // dims num
      buffer_size +=
          sizeof(uint32_t) * node->input_tensors[j]->attr->dims.size();  // dims
      buffer_size += sizeof(PrecisionType);   // percision type
      buffer_size += sizeof(DataLayoutType);  // layout type
      buffer_size += sizeof(uint32_t);        // zero point
      buffer_size += sizeof(float);           // scale
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->input_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->input_tensors[j]->attr->dims[k];
      }
      buffer_size += tensor_size;
    }
    buffer_size += sizeof(uint32_t);  // node's output-tensor num
    for (int j = 0; j < node->output_tensors.size(); j++) {
      buffer_size += sizeof(uint32_t);  // dims num
      buffer_size +=
          sizeof(uint32_t) * node->input_tensors[j]->attr->dims.size();  // dims
      buffer_size += sizeof(PrecisionType);   // percision type
      buffer_size += sizeof(DataLayoutType);  // layout type
      buffer_size += sizeof(uint32_t);        // zero point
      buffer_size += sizeof(float);           // scale
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->output_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->output_tensors[j]->attr->dims[k];
      }
      buffer_size += tensor_size;
    }
  }
  return buffer_size;
  // Only need serialize op's info&tensor, graph's tensor will init in
  // func:SetInputsOutputs
}

void SerializeModelbuffer(fakedevice_nn_graph_t *graph, void *buffer) {
  int node_num = graph->node_table.size();
  // node num
  *reinterpret_cast<uint32_t *>(buffer) = graph->node_table.size();
  buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
  for (int i = 0; i < node_num; i++) {
    fakedevice_nn_node_t *node = graph->node_table[i];
    // op type
    *reinterpret_cast<fakedevice_nn_op_t *>(buffer) = node->op;
    buffer = reinterpret_cast<fakedevice_nn_op_t *>(buffer) + 1;
    // op param
    memcpy(reinterpret_cast<fakedevice_nn_param_t *>(buffer),
           &(node->nn_param.conv2d_param),
           sizeof(fakedevice_nn_param_t));
    buffer =
        reinterpret_cast<uint8_t *>(buffer) + sizeof(fakedevice_nn_param_t);
    // input tensor number
    *reinterpret_cast<uint32_t *>(buffer) = node->input_tensors.size();
    buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
    for (int j = 0; j < node->input_tensors.size(); j++) {
      // dims num
      *reinterpret_cast<uint32_t *>(buffer) =
          node->input_tensors[j]->attr->dims.size();
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // dims
      for (int k = 0; k < node->input_tensors[j]->attr->dims.size(); k++) {
        *reinterpret_cast<uint32_t *>(buffer) =
            node->input_tensors[j]->attr->dims[k];
        buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      }
      // percision type
      *reinterpret_cast<PrecisionType *>(buffer) =
          node->input_tensors[j]->attr->precision;
      buffer = reinterpret_cast<PrecisionType *>(buffer) + 1;
      // layout typr
      *reinterpret_cast<DataLayoutType *>(buffer) =
          node->input_tensors[j]->attr->layout;
      buffer = reinterpret_cast<DataLayoutType *>(buffer) + 1;
      // zero point
      *reinterpret_cast<uint32_t *>(buffer) =
          node->input_tensors[j]->attr->qntParamAffineAsymmetric.zero_point[0];
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // scale
      *reinterpret_cast<float *>(buffer) =
          node->input_tensors[j]->attr->qntParamAffineAsymmetric.scale[0];
      buffer = reinterpret_cast<float *>(buffer) + 1;
      // tensor data
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->input_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->input_tensors[j]->attr->dims[k];
      }
      memcpy(reinterpret_cast<uint8_t *>(buffer),
             reinterpret_cast<uint8_t *>(node->input_tensors[j]->data),
             sizeof(tensor_size));
      buffer = reinterpret_cast<uint8_t *>(buffer) +
               sizeof(uint8_t) * tensor_size;  // data type is uint8
    }
    // output tensor number
    *reinterpret_cast<uint32_t *>(buffer) = node->output_tensors.size();
    buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
    for (int j = 0; j < node->output_tensors.size(); j++) {
      // dims num
      *reinterpret_cast<uint32_t *>(buffer) =
          node->output_tensors[j]->attr->dims.size();
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // dims
      for (int k = 0; k < node->output_tensors[j]->attr->dims.size(); k++) {
        *reinterpret_cast<uint32_t *>(buffer) =
            node->output_tensors[j]->attr->dims[k];
        buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      }
      // percision type
      *reinterpret_cast<PrecisionType *>(buffer) =
          node->output_tensors[j]->attr->precision;
      buffer = reinterpret_cast<PrecisionType *>(buffer) + 1;
      // layout typr
      *reinterpret_cast<DataLayoutType *>(buffer) =
          node->output_tensors[j]->attr->layout;
      buffer = reinterpret_cast<DataLayoutType *>(buffer) + 1;
      // zero point
      *reinterpret_cast<uint32_t *>(buffer) =
          node->output_tensors[j]->attr->qntParamAffineAsymmetric.zero_point[0];
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // scale
      *reinterpret_cast<float *>(buffer) =
          node->output_tensors[j]->attr->qntParamAffineAsymmetric.scale[0];
      buffer = reinterpret_cast<float *>(buffer) + 1;
      // tensor data
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->output_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->output_tensors[j]->attr->dims[k];
      }
      memcpy(reinterpret_cast<uint8_t *>(buffer),
             reinterpret_cast<uint8_t *>(node->output_tensors[j]->data),
             sizeof(tensor_size));
      buffer = reinterpret_cast<uint8_t *>(buffer) +
               sizeof(uint8_t) * tensor_size;  // data type is uint8;
    }
  }
}

void DeserializeModelbuffer(fakedevice_nn_graph_t *graph, void *buffer) {
  // node num
  int node_num = *reinterpret_cast<uint32_t *>(buffer);
  buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
  for (int i = 0; i < node_num; i++) {
    fakedevice_nn_node_t *node = new fakedevice_nn_node_t();
    graph->node_table.push_back(node);
    // op type
    node->op = *reinterpret_cast<fakedevice_nn_op_t *>(buffer);
    buffer = reinterpret_cast<fakedevice_nn_op_t *>(buffer) + 1;
    // op param
    memcpy(&(node->nn_param.conv2d_param),
           reinterpret_cast<fakedevice_nn_param_t *>(buffer),
           sizeof(fakedevice_nn_param_t));
    buffer =
        reinterpret_cast<uint8_t *>(buffer) + sizeof(fakedevice_nn_param_t);
    // input tensor number
    int input_tensor_num = *reinterpret_cast<uint32_t *>(buffer);
    buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
    for (int j = 0; j < input_tensor_num; j++) {
      std::shared_ptr<fakedevice_nn_tensor_t> tensor =
          std::make_shared<fakedevice_nn_tensor_t>();
      graph->node_table[i]->input_tensors.push_back(tensor.get());
      // dims num
      int dims_num = *reinterpret_cast<uint32_t *>(buffer);
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      auto attr = std::make_shared<fake_ddk::nn::TensorAttr>();
      graph->node_table[i]->input_tensors[j]->attr = attr;
      // dims
      for (int k = 0; k < dims_num; k++) {
        uint32_t dims = *reinterpret_cast<uint32_t *>(buffer);
        attr->dims.push_back(dims);
        buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      }
      // percision type
      attr->precision = *reinterpret_cast<PrecisionType *>(buffer);
      buffer = reinterpret_cast<PrecisionType *>(buffer) + 1;
      // layout type
      attr->layout = *reinterpret_cast<DataLayoutType *>(buffer);
      buffer = reinterpret_cast<DataLayoutType *>(buffer) + 1;
      // zero point
      attr->qntParamAffineAsymmetric.zero_point[0] =
          *reinterpret_cast<uint32_t *>(buffer);
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // scale
      attr->qntParamAffineAsymmetric.scale[0] =
          *reinterpret_cast<float *>(buffer);
      buffer = reinterpret_cast<float *>(buffer) + 1;
      // tensor data
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->input_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->input_tensors[j]->attr->dims[k];
      }
      node->input_tensors[j]->data =
          malloc(sizeof(tensor_size));  // data type is uint8
      memcpy(reinterpret_cast<uint8_t *>(node->input_tensors[j]->data),
             reinterpret_cast<uint8_t *>(buffer),
             sizeof(tensor_size));
      buffer =
          reinterpret_cast<uint8_t *>(buffer) + sizeof(uint8_t) * tensor_size;
    }
    // output tensor number
    int output_tensor_num = *reinterpret_cast<uint32_t *>(buffer);
    buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
    for (int j = 0; j < output_tensor_num; j++) {
      std::shared_ptr<fakedevice_nn_tensor_t> tensor =
          std::make_shared<fakedevice_nn_tensor_t>();
      graph->node_table[i]->output_tensors.push_back(tensor.get());
      // dims num
      int dims_num = *reinterpret_cast<uint32_t *>(buffer);
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      auto attr = std::make_shared<fake_ddk::nn::TensorAttr>();
      graph->node_table[i]->output_tensors[j]->attr = attr;
      // dims
      for (int k = 0; k < dims_num; k++) {
        uint32_t dims = *reinterpret_cast<uint32_t *>(buffer);
        attr->dims.push_back(dims);
        buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      }
      // percision type
      attr->precision = *reinterpret_cast<PrecisionType *>(buffer);
      buffer = reinterpret_cast<PrecisionType *>(buffer) + 1;
      // layout type
      attr->layout = *reinterpret_cast<DataLayoutType *>(buffer);
      buffer = reinterpret_cast<DataLayoutType *>(buffer) + 1;
      // zero point
      attr->qntParamAffineAsymmetric.zero_point[0] =
          *reinterpret_cast<uint32_t *>(buffer);
      buffer = reinterpret_cast<uint32_t *>(buffer) + 1;
      // scale
      attr->qntParamAffineAsymmetric.scale[0] =
          *reinterpret_cast<float *>(buffer);
      buffer = reinterpret_cast<float *>(buffer) + 1;
      // tensor data
      uint32_t tensor_size = 1;
      for (int k = 0; k < node->output_tensors[j]->attr->dims.size(); k++) {
        tensor_size *= node->output_tensors[j]->attr->dims[k];
      }
      node->output_tensors[j]->data =
          malloc(sizeof(tensor_size));  // data type is uint8
      memcpy(reinterpret_cast<uint8_t *>(node->output_tensors[j]->data),
             reinterpret_cast<uint8_t *>(buffer),
             sizeof(tensor_size));
      buffer =
          reinterpret_cast<uint8_t *>(buffer) + sizeof(uint8_t) * tensor_size;
    }
  }
}
}  // namespace nn
}  // namespace fake_ddk
