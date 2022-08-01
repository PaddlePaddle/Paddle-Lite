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
#include "fake_ddk/micros.h"

namespace fake_ddk {

typedef struct {
  /* The index of the input or output tensors of the graph */
  int index;
  /* The shapes of the input and output tensors for the current execution is
   * mainly used in dynamic shape scenarios. By default, it is equal to the
   * shapes of the input and output tensors when building the graph. */
  std::vector<int32_t> shape;
  void* buffer;
} Argument;

/* Build a device program from a graph and create a execution for inference
 *  You need a graph before you create a execution.
 *  The overall process is as follows:
 *  <pre>
 *  graph = new Graph();
 *  ...
 *  exector = new Execution(graph);
 *  exector->Build();
 *
 *  If no errors are encountered, a program corresponding to the device will be
 * created.
 *  In this way, you can start inference on the device.
 *
 *  while(something) {
 *    exector->SetInputs(inputs);
 *    exector->Run();
 *    exector->GetOutputs(outputs);
 *  }
 * </pre>
*/
class FAKE_DDK_EXPORT Execution {
 public:
  /* Create a execution from a graph.
  */
  explicit Execution(Graph* graph);
  ~Execution();

  /* Build a device program from a graph.
   * @return StatusType::SUCCESS when success.
   */
  int Build();
  /* Build a device program from a graph, and serialize it to a buffer.
   * @return StatusType::SUCCESS when success.
   */
  int Build(std::vector<uint8_t>* buffer);

  /* Set the input arguments before starting the execution.
   * When using a quantitative model, the input data must also be the quantized
   * data.
   * When calling SetInputs() multiple times in succession, only the last data
   * takes effect.
   * @param inputs [in] input data
   * @return StatusType::SUCCESS when success.
   */
  int SetInputs(const std::vector<Argument>& input_arguments);

  /* Start the execution and wait for it to finish synchronously.
   * This function should be called after the SetInputs() function.
   * @return StatusType::SUCCESS when success.
   */
  int Run();

  /* Get the output arguments after the execution is finished.
   * This function should be called after the Run() function.
   * If using a quantitative model, the output is quantized data
   * @return StatusType::SUCCESS when success.
   */
  int GetOutputs(std::vector<Argument>* output_arguments);

 private:
  Graph* graph_;
  /* The sorted operators in topological order */
  std::vector<Operator*> operators_;
};

}  // namespace fake_ddk
