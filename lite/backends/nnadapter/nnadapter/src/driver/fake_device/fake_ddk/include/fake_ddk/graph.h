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

#include <memory>
#include <string>
#include <vector>
#include "fake_ddk/operator.h"
#include "fake_ddk/tensor.h"

namespace fake_ddk {
namespace nn {

typedef struct _fakedevice_nn_node {
  /*
   * Operation type
   * @see fakedevice_nn_op_t
   */
  fakedevice_nn_op_t op;
  /* Node inputs */
  std::vector<fakedevice_nn_tensor_t*> input_tensors;
  /* Node outputs */
  std::vector<fakedevice_nn_tensor_t*> output_tensors;
  /* Operation parameters */
  fakedevice_nn_param_t nn_param;
} fakedevice_nn_node_t;

typedef struct _fakedevice_nn_graph {
  /* Tensor number */
  uint32_t tensor_num;

  /* Node list of this graph */
  /* Node table */
  std::vector<fakedevice_nn_node_t*> node_table;
  /* Inputs to the graph */
  std::vector<fakedevice_nn_tensor_t*> input_tensors;
  /* Outputs to the graph */
  std::vector<fakedevice_nn_tensor_t*> output_tensors;
} fakedevice_nn_graph_t;

typedef struct _fakedevice_model_buffer {
  int length;
  void* data;
} fakedevice_model_buffer;

/* Graph is used to create and save tensor and Operator, and the connection
 * relationship of these Operators.
 * It is mainly used to save model information and is not actually created on
 * the FAKE_DEVICE.
*/
class Graph {
  friend class Exection;

 private:
  void* fakedevice_graph_;
  fakedevice_model_buffer* fm_;

 public:
  Graph();
  explicit Graph(char* cache_path);
  ~Graph();
  /* Add an Operator.
   * @param type [in] Operator Type
   * @param inputs [in] input tensors
   * @param outputs [in] outputs tensors
   * @param attrs [in] attributes of Operator
   * @param name [in] Operator's name
   * @return The corresponding Operator is returned on success, otherwise it
   * returns nullptr
  */
  int AddOperator(OperatorType type,
                  std::vector<std::shared_ptr<Tensor>> inputs,
                  std::vector<std::shared_ptr<Tensor>> outputs,
                  void* attrs,
                  std::string name = "");
  /* Create a tensor.
   * @param attr [in] attributes of tensor, cannot be empty.
   * @param data [in] tensor data, can be empty.
   * @return the pointer of tensor
  */
  std::shared_ptr<Tensor> CreateTensor(std::shared_ptr<TensorAttr> attr,
                                       void* data);

  /* Set the input and output tensor of the graph.
   * @param input_tensors [in] input tensors
   * @param output_tensors [in] output tensors
   * @return FAKE_DDK_SUCCESS when success
  */
  int SetInputsOutputs(std::vector<std::shared_ptr<Tensor>> input_tensors,
                       std::vector<std::shared_ptr<Tensor>> output_tensors);
  /* Enable model cache.
   * @return FAKE_DDK_SUCCESS when the ddk support it
  */
  int EnableCache();

  int DisableCache();
  /* Load model cache. Deserialize cache buffer to fake_ddk's grpah
   * @return FAKE_DDK_SUCCESS when the ddk support it
  */
  int LoadCache(char* cache_buffer, int size);
};
}  // namespace nn
}  // namespace fake_ddk
