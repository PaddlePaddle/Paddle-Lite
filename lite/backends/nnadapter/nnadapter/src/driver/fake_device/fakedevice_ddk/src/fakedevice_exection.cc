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
#include "fakedevice/fakedevice_pub.h"

namespace fakedevice {
namespace nn {

#ifndef FALSE   /* in case these macros already exist */
#define FALSE 0 /* values of boolean */
#endif
#ifndef TRUE
#define TRUE 1
#endif
extern int conv_uint8_fp32_mmad(fakedevice_nn_tensor_t* input_tensor,
                                fakedevice_nn_tensor_t* output_tensor,
                                fakedevice_nn_tensor_t* kernel,
                                fakedevice_nn_tensor_t* bias,
                                fakedevice_nn_conv2d_param* conv_param);
extern int conv_uint8_8bit_mmad(fakedevice_nn_tensor_t* input_tensor,
                                fakedevice_nn_tensor_t* output_tensor,
                                fakedevice_nn_tensor_t* kernel,
                                fakedevice_nn_tensor_t* bias,
                                fakedevice_nn_conv2d_param* conv_param);
extern int CalculateModelbufferSize(fakedevice_nn_graph_t* graph);
extern void SerializeModelbuffer(fakedevice_nn_graph_t* graph, void* buffer);
Exection::Exection(Graph* graph) {
  printf("comein Exection\n ");
  graph_ = graph;
}
Exection::~Exection() {}

/************************************************
*   setup graph and optimize graph
************************************************/
int Exection::Build() {
  printf("comein Build()\n ");
  return FAKE_DEVICE_SUCCESS;
}

int Exection::Build(fakedevice_model_buffer* fm) {
  printf("comein Build(fm)\n ");
  printf("NEED model_cache\n ");

  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);
  int buffer_length = CalculateModelbufferSize(graph);
  void* buffer = malloc(buffer_length * sizeof(uint8_t));
  SerializeModelbuffer(graph, buffer);
  fm->length = buffer_length;
  fm->data = buffer;
  return FAKE_DEVICE_SUCCESS;
}

/** Set model inputs.
 *  When using a quantitative model, the input data must also be the quantized
 * data.
 *  When calling SetInputs() multiple times in succession, only the last data
 * takes effect.
 *
 *  @param inputs [in] input data
 *  @return FAKE_DEVICE_SUCCESS when success
 */
int Exection::SetInputs(std::vector<InputInfo> inputs) {
  printf("comein SetInputs\n ");
  int i;
  // printf("\n ddk SetInputs in,and graph_'s addr is %x\n ", graph_);
  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);

  // printf("type is %d\n",inputs[i].type);
  for (i = 0; i < graph->input_tensors.size(); i++) {
    printf("comein SetInputs 1\n ");
    printf("graph->input_tensors[i]->data ===== %x, inputs[i].buf =====%x \n",
           graph->input_tensors[i]->data,
           inputs[i].buf);
    memcpy(graph->input_tensors[i]->data, inputs[i].buf, inputs[i].size);
    printf("comein SetInputs 2\n ");
  }
  return FAKE_DEVICE_SUCCESS;
}

/** Do inference on fake_device .
 *  This function should be called after the SetInputs() function.
 *
 *  @return FAKE_DEVICE_SUCCESS when success
 */
int Exection::Run() {
  printf("comein Run()\n ");
  int status;
  // printf("begin to run\n");
  fakedevice_nn_graph_t* graph =
      static_cast<fakedevice_nn_graph_t*>(graph_->fakedevice_graph_);
  printf("Exection::Run() :graph_ = %x \n", graph_);
  printf("Exection::Run() :graph_->fakedevice_graph_ = %x \n",
         graph_->fakedevice_graph_);
  for (int i = 0; i < graph->node_table.size(); i++) {
    fakedevice_nn_node_t* node = graph->node_table[i];
    switch (node->op) {
      case fakedevice::nn::OperatorType::FAKE_DEVICE_CONV2D:
        if (i == 0) {  // first node
          status = conv_uint8_fp32_mmad(graph->input_tensors[0],
                                        node->output_tensors[0],
                                        node->input_tensors[1],
                                        node->input_tensors[2],
                                        &(node->nn_param.conv2d_param));
        } else if (i == graph->node_table.size() - 1) {  // last node
          status = conv_uint8_fp32_mmad(node->input_tensors[0],
                                        graph->output_tensors[0],
                                        node->input_tensors[1],
                                        node->input_tensors[2],
                                        &(node->nn_param.conv2d_param));
        } else {
          status = conv_uint8_fp32_mmad(node->input_tensors[0],
                                        node->output_tensors[0],
                                        node->input_tensors[1],
                                        node->input_tensors[2],
                                        &(node->nn_param.conv2d_param));
        }
        break;
      default:
        printf("this operator not support yet\n");
        break;
    }
  }
  if (status != FAKE_DEVICE_SUCCESS) {
    printf("process graph fail\n");
    return FAKE_DEVICE_FAILURE;
  }
  return FAKE_DEVICE_SUCCESS;
}

/** Get model outputs.
 *  This function should be called after the Run() function.
 *  If using a quantitative model, the output is quantized data
 *
 *  @return FAKE_DEVICE_SUCCESS when success
 */
int Exection::GetOutputs(std::vector<OutputInfo> outputs) {
  printf("comein GetOutputs\n ");
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
    // for debug, dump ddk output
    /*
    for (j = 0; j < 10; j++) {
      printf("ddk-output<<[%d:%d]\n", j,
    (reinterpret_cast<uint8_t*>((tensor->data)[j])));
    }
    */
  }
  return FAKE_DEVICE_SUCCESS;
}

}  // namespace nn
}  // namespace fakedevice
