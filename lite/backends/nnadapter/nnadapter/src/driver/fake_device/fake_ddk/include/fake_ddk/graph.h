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

#include <list>
#include <vector>
#include "fake_ddk/micros.h"
#include "fake_ddk/operator.h"
#include "fake_ddk/tensor.h"

namespace fake_ddk {

/* A graph is mainly used to represent a neural network model, it contains
 * operators and tensors and their dependencies.
*/
class FAKE_DDK_EXPORT Graph {
  friend class Execution;

 public:
  /* Create a new graph */
  Graph();
  /* Restore a graph from a buffer */
  explicit Graph(const std::vector<uint8_t>& buffer);
  ~Graph();

  /* Reset the graph, clear the tensors and operators */
  void Clear();

  /* Create and add a new tensor, and set its data if provided.
   * @param attr [in] the attributes of tensor, cannot be empty.
   * @param data [in] the data of constant/weight tensor, can be empty.
   * @return the pointer of tensor.
  */
  Tensor* AddTensor(const TensorAttr& attr, void* data);

  /* Create and add a new operator, and set its input and output tensors.
   * @param type [in] the operator type.
   * @param inputs [in] the input tensors.
   * @param outputs [in] the outputs tensors.
   * @param attr [in] the operator attributes.
   * @return the corresponding operator is returned on success, otherwise it
   * returns nullptr.
  */
  Operator* AddOperator(OperatorType type,
                        const std::vector<Tensor*>& input_tensors,
                        const std::vector<Tensor*>& output_tensors,
                        void* attr);

  /* Identify the input and output tensors of the graph.
   * @param input_tensors [in] the input tensors.
   * @param output_tensors [in] the output tensors.
   * @return StatusType::SUCCESS when success.
  */
  int IdentifyInputsAndOutputs(const std::vector<Tensor*>& input_tensors,
                               const std::vector<Tensor*>& output_tensors);

  /** Query the attributes of input and output tensors of the graph.
    * @param input_tensors [out] the attributes of the input tensors.
    * @param output_tensors [out] the attributes of the output tensors.
    * @return StatusType::SUCCESS when success.
  */
  int QueryInputsAndOutputs(std::vector<TensorAttr>* input_attrs,
                            std::vector<TensorAttr>* output_attrs);

 public:
  /* The tensors and operators of the graph */
  std::list<Tensor> tensors_;
  std::list<Operator> operators_;
  /* The input tensors of the graph */
  std::vector<Tensor*> input_tensors_;
  /* The output tensors of the graph */
  std::vector<Tensor*> output_tensors_;
};

}  // namespace fake_ddk
