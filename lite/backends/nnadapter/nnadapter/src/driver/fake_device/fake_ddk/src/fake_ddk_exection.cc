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

#ifndef FALSE   /* In case these macros already exist */
#define FALSE 0 /* Values of boolean */
#endif
#ifndef TRUE
#define TRUE 1
#endif
extern int conv_uint8_compute(fakedevice_nn_tensor_t* input_tensor,
                              fakedevice_nn_tensor_t* output_tensor,
                              fakedevice_nn_tensor_t* kernel,
                              fakedevice_nn_tensor_t* bias,
                              fakedevice_nn_conv2d_param* conv_param);
extern int CalculateModelbufferSize(fakedevice_nn_graph_t* graph);
extern void SerializeModelbuffer(fakedevice_nn_graph_t* graph, void* buffer);
Exection::Exection(Graph* graph) {
  fprintf(stderr, "fake_ddk: Exection\n");
  graph_ = graph;
}
Exection::~Exection() {}

/* setup graph and optimize graph */
int Exection::Build() {
  fprintf(stderr, "fake_ddk: Build()\n");
  return FAKE_DDK_SUCCESS;
}

int Exection::Build(fakedevice_model_buffer* fm) {
  fprintf(stderr, "fake_ddk: NEED model_cache\n");

  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);
  int buffer_length = CalculateModelbufferSize(graph);
  void* buffer = malloc(buffer_length * sizeof(uint8_t));
  SerializeModelbuffer(graph, buffer);
  fm->length = buffer_length;
  fm->data = buffer;
  return FAKE_DDK_SUCCESS;
}

/* Set model inputs.
 * When using a quantitative model, the input data must also be the quantized
 * data.
 * When calling SetInputs() multiple times in succession, only the last data
 * takes effect.
 *
 * @param inputs [in] input data
 * @return FAKE_DDK_SUCCESS when success
 */
int Exection::SetInputs(std::vector<InputInfo> inputs) {
  fprintf(stderr, "fake_ddk: SetInputs\n");
  int i;
  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);
  for (i = 0; i < graph->input_tensors.size(); i++) {
    memcpy(graph->input_tensors[i]->data, inputs[i].buf, inputs[i].size);
  }
  return FAKE_DDK_SUCCESS;
}

/* Do inference on fake_device .
 * This function should be called after the SetInputs() function.
 *
 * @return FAKE_DDK_SUCCESS when success
 */
int Exection::Run() {
  fprintf(stderr, "fake_ddk: Run()\n");
  int status;
  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);
  for (int i = 0; i < graph->node_table.size(); i++) {
    fakedevice_nn_node_t* node = graph->node_table[i];
    switch (node->op) {
      case fake_ddk::nn::OperatorType::FAKE_DEVICE_CONV2D:
        if (i == 0) {  // first node
          status = conv_uint8_compute(graph->input_tensors[0],
                                      node->output_tensors[0],
                                      node->input_tensors[1],
                                      node->input_tensors[2],
                                      &(node->nn_param.conv2d_param));
        } else if (i == graph->node_table.size() - 1) {  // last node
          status = conv_uint8_compute(node->input_tensors[0],
                                      graph->output_tensors[0],
                                      node->input_tensors[1],
                                      node->input_tensors[2],
                                      &(node->nn_param.conv2d_param));
        } else {
          status = conv_uint8_compute(node->input_tensors[0],
                                      node->output_tensors[0],
                                      node->input_tensors[1],
                                      node->input_tensors[2],
                                      &(node->nn_param.conv2d_param));
        }
        break;
      default:
        fprintf(stderr, "fake_ddk: this operator not support yet\n");
        break;
    }
  }
  if (status != FAKE_DDK_SUCCESS) {
    fprintf(stderr, "fake_ddk: process graph fail\n");
    return FAKE_DDK_FAILURE;
  }
  return FAKE_DDK_SUCCESS;
}

/* Get model outputs.
 *  This function should be called after the Run() function.
 *  If using a quantitative model, the output is quantized data
 *
 *  @return FAKE_DDK_SUCCESS when success
 */
int Exection::GetOutputs(std::vector<OutputInfo> outputs) {
  fprintf(stderr, "fake_ddk: GetOutputs\n");
  int i, j;
  fakedevice_nn_tensor_t* tensor;
  uint32_t sz;
  float* buffer = NULL;
  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);

  for (i = 0; i < static_cast<int>(graph->output_tensors.size()); i++) {
    tensor = graph->output_tensors[i];
    sz = 1;
    for (j = 0; j < static_cast<int>(tensor->attr->dims.size()); j++) {
      sz *= tensor->attr->dims[j];
    }
    memcpy(outputs[i].buf, tensor->data, sizeof(uint8_t) * sz);
    /*
    for (j = 0; j < 10; j++) {
      fprintf(stderr, "fake_ddk: ddk-output<<[%d:%d]\n", j,
    (reinterpret_cast<uint8_t*>((tensor->data)[j])));
    }
    */
  }
  return FAKE_DDK_SUCCESS;
}

}  // namespace nn
}  // namespace fake_ddk
