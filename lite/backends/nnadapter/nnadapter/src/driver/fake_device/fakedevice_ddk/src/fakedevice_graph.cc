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

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include "fakedevice/fakedevice_pub.h"

namespace fakedevice {
namespace nn {

extern void DeserializeModelbuffer(fakedevice_nn_graph_t *graph, void *buffer);
static fakedevice_nn_graph_t *g_fakedevice_graph;
unsigned char *g_outputbuf = NULL;

static fakedevice_nn_node_t *fakedevice_nn_AddNode(fakedevice_nn_graph_t *graph,
                                                   fakedevice_nn_op_t optype) {
  printf("comein fakedevice_nn_AddNode\n ");
  fakedevice_nn_node_t *node_ = new fakedevice_nn_node_t();
  if (NULL != node_) {
    graph->node_table.push_back(node_);
    node_->op = optype;
    printf("comein fakedevice_nn_AddNode finished\n ");
    return node_;
  } else {
    printf("fake device nn: AddNode FAIL ! \n");
    return NULL;
  }
}

std::shared_ptr<fakedevice_nn_tensor_t> Graph::CreateTensor(
    std::shared_ptr<TensorAttr> attr, void *data) {
  printf("comein CreateTensor\n ");
  std::shared_ptr<fakedevice_nn_tensor_t> tensor_ =
      std::make_shared<fakedevice_nn_tensor_t>();
  if (NULL != tensor_) {
    tensor_->attr = attr;
    tensor_->data = data;  // if need re-arrange, alloc a new buffer
    printf("tensor_->data ===== %x \n", tensor_->data);
    printf("comein CreateTensor and finish copy\n ");
  } else {
    printf("fake device nn: CreateTensor FAIL ! \n");
    return NULL;
  }

  return tensor_;
}

static int AddOperatorConv2d(
    std::vector<std::shared_ptr<fakedevice_nn_tensor_t>> inputs,
    std::vector<std::shared_ptr<fakedevice_nn_tensor_t>> outputs,
    void *attrs) {
  printf("comein addOperator_CONV2D\n ");
  fakedevice_nn_node_t *node;
  auto *attr = static_cast<Conv2DAttr *>(attrs);
  int i = 0;
  node = fakedevice_nn_AddNode(g_fakedevice_graph, FAKE_DEVICE_CONV2D);

  if (node == nullptr) {
    printf("add node is fail!\n");
    return -1;
  }
  printf("comein addOperator_CONV2D 1\n ");
  node->nn_param.conv2d_param.ksize[0] = attr->ksize[0];
  node->nn_param.conv2d_param.ksize[1] = attr->ksize[1];
  node->nn_param.conv2d_param.weights = attr->weights;
  node->nn_param.conv2d_param.stride[0] = attr->stride[0];
  node->nn_param.conv2d_param.stride[1] = attr->stride[1];
  node->nn_param.conv2d_param.pad[0] = attr->pad[0];
  node->nn_param.conv2d_param.pad[1] = attr->pad[1];
  node->nn_param.conv2d_param.pad[2] = attr->pad[2];
  node->nn_param.conv2d_param.pad[3] = attr->pad[3];
  node->nn_param.conv2d_param.group = attr->group;
  node->nn_param.conv2d_param.dilation[0] = attr->dilation[0];
  node->nn_param.conv2d_param.dilation[1] = attr->dilation[1];
  node->nn_param.conv2d_param.multiplier = attr->multiplier;
  node->nn_param.conv2d_param.has_relu = attr->has_relu;
  printf("comein addOperator_CONV2D 2\n ");
  node->input_tensors.push_back(inputs[0].get());
  node->input_tensors.push_back(inputs[1].get()); /* data_weight */
  node->input_tensors.push_back(inputs[2].get()); /* data_bias */
  printf("comein addOperator_CONV2D 3\n ");
  for (i = 0; i < static_cast<int>(outputs.size()); ++i) {
    node->output_tensors.push_back(outputs[i].get());
  }
  // node->output.tensors[i] = output_tensors[i];
  printf("comein addOperator_CONV2D finished\n ");
  return FAKE_DEVICE_SUCCESS;
}

int AddOperatorRelu(std::vector<std::shared_ptr<Tensor>> inputs,
                    std::vector<std::shared_ptr<Tensor>> outputs,
                    void *attrs) {
  return 0;
}

int Graph::AddOperator(OperatorType type,
                       std::vector<std::shared_ptr<Tensor>> inputs,
                       std::vector<std::shared_ptr<Tensor>> outputs,
                       void *attrs,
                       std::string name) {
  printf("comein AddOperator\n ");
  switch (type) {
    case fakedevice::nn::OperatorType::FAKE_DEVICE_CONV2D:
      AddOperatorConv2d(inputs, outputs, attrs);
      break;
    case fakedevice::nn::OperatorType::FAKE_DEVICE_RELU:
      AddOperatorRelu(inputs, outputs, attrs);
      break;
    default:
      printf("this operator not support yet\n");
      return FAKE_DEVICE_FAILURE;
  }
  return FAKE_DEVICE_SUCCESS;
}

int Graph::SetInputsOutputs(
    std::vector<std::shared_ptr<Tensor>> input_tensors,
    std::vector<std::shared_ptr<Tensor>> output_tensors) {
  printf("comein SetInputsOutputs\n ");
  int i;
  fakedevice_nn_graph_t *graph;
  graph = static_cast<fakedevice_nn_graph_t *>(fakedevice_graph_);

  // fakedevice_nn_SetGraphInputs(graph, NULL, (uint32_t)input_tensors.size());
  // fakedevice_nn_SetGraphOutputs(graph, NULL,
  // (uint32_t)output_tensors.size());
  for (i = 0; i < static_cast<int>(input_tensors.size()); i++) {
    graph->input_tensors.push_back(input_tensors[i].get());
    printf("SetInputsOutputs graph->input_tensors[i].data ===== %x \n",
           graph->input_tensors[i]->data);
    printf(" graph->input_tensors[i]->attr->dims[0] = %d \n",
           graph->input_tensors[i]->attr->dims[0]);
    printf(" graph->input_tensors[i]->attr->dims[1] = %d \n",
           graph->input_tensors[i]->attr->dims[1]);
    printf(" graph->input_tensors[i]->attr->dims[2] = %d \n",
           graph->input_tensors[i]->attr->dims[2]);
    printf(" graph->input_tensors[i]->attr->dims[3] = %d \n",
           graph->input_tensors[i]->attr->dims[3]);
    int input_size = 1;
    for (int k = 0; k < graph->input_tensors[i]->attr->dims.size(); k++) {
      input_size *= graph->input_tensors[i]->attr->dims[k];
    }
    graph->input_tensors[i]->data =
        reinterpret_cast < void * (malloc(input_size * sizeof(uint8_t)));
    printf("malloc input Tensor  graph->output_tensors[i].data = %x \n",
           graph->input_tensors[i]->data);
  }
  for (i = 0; i < static_cast<int>(output_tensors.size()); i++) {
    graph->output_tensors.push_back(output_tensors[i].get());
    printf("SetInputsOutputs graph->output_tensors[i].data = %x \n",
           graph->output_tensors[i]->data);
    int output_size = 1;
    for (int k = 0; k < graph->output_tensors[i]->attr->dims.size(); k++) {
      output_size *= graph->output_tensors[i]->attr->dims[k];
    }
    graph->output_tensors[i]->data =
        reinterpret_cast<void *>(malloc(output_size * sizeof(uint8_t)));
    printf("malloc output Tensor  graph->output_tensors[i].data = %x \n",
           graph->output_tensors[i]->data);
  }
  return FAKE_DEVICE_SUCCESS;
}

int Graph::EnableCache() {
  int ret = FAKE_DEVICE_SUCCESS;
  return FAKE_DEVICE_SUCCESS;
}

int Graph::DisableCache() {
  int ret = FAKE_DEVICE_SUCCESS;
  return ret;
}

int Graph::LoadCache(char *cache_buffer, int size) {
  int ret = FAKE_DEVICE_SUCCESS;
  DeserializeModelbuffer(
      reinterpret_cast<fakedevice_nn_graph_t *>(fakedevice_graph_),
      reinterpret_cast<void *>(cache_buffer));
  return ret;
}

Graph::Graph() {
  printf("comein Graph()\n ");
  fakedevice_graph_ = new fakedevice_nn_graph_t();
  if ((nullptr == fakedevice_graph_)) {
    printf("create graph is fail!\n");
  }
  memset(fakedevice_graph_, 0, sizeof(fakedevice_nn_graph_t));
  g_fakedevice_graph = static_cast<fakedevice_nn_graph_t *>(fakedevice_graph_);
}

Graph::~Graph() {}

}  // namespace nn
}  // namespace fakedevice
