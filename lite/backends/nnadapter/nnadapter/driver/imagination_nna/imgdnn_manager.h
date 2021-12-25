// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <imgdnn.h>
#include <unistd.h>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utility/logging.h"

namespace nnadapter {
namespace imagination_nna {

#define IMGDNN_CHECK_SUCCESS(error_code_)           \
  NNADAPTER_CHECK_EQ(static_cast<int>(error_code_), \
                     static_cast<int>(IMGDNN_SUCCESS))

class ImgdnnManager {
 public:
  ImgdnnManager();
  virtual ~ImgdnnManager();
  uint8_t *GetBuffer(size_t size);
  imgdnn_network GetNetwork() { return network_; }
  imgdnn_tensor CreateInputTensor(imgdnn_tensor_descriptor *desc);
  imgdnn_tensor CreateFixedInputTensor(imgdnn_tensor_descriptor *desc,
                                       const void *const data,
                                       bool copy);

  // Layers
  imgdnn_tensor CreateConvolutionLayer(imgdnn_tensor input_tensor,
                                       imgdnn_tensor weights_tensor,
                                       imgdnn_tensor bias_tensor,
                                       imgdnn_quant_param quant_param,
                                       unsigned int stride[2],
                                       unsigned int pad_begin[2],
                                       unsigned int pad_end[2],
                                       unsigned int dilation[2],
                                       bool use_dwconv = false);
  imgdnn_tensor CreateBatchNormLayer(imgdnn_tensor input_tensor,
                                     const void *const mean_data,
                                     const void *const variance_data,
                                     const float eps);
  imgdnn_tensor CreatePoolingLayer(imgdnn_tensor input_tensor,
                                   imgdnn_quant_param quant_param,
                                   const unsigned int size[2],
                                   const unsigned int stride[2],
                                   const unsigned int pad_to_begin[2],
                                   const unsigned int pad_to_end[2],
                                   bool count_include_pad,
                                   imgdnn_pooling_type type);
  imgdnn_tensor CreateFullyConnectedLayer(imgdnn_tensor input_tensor,
                                          imgdnn_tensor weights_tensor,
                                          imgdnn_tensor bias_tensor,
                                          imgdnn_quant_param quant_param);
  imgdnn_tensor CreateMatMulLayer(imgdnn_tensor input0_tensor,
                                  imgdnn_tensor input1_tensor,
                                  imgdnn_quant_param quant_param);
  imgdnn_tensor CreateElementwiseOpsLayer(
      imgdnn_tensor input0_tensor,
      imgdnn_tensor input1_tensor,
      imgdnn_operation_binary imgdnn_operation,
      imgdnn_quant_param quant_param);
  imgdnn_tensor CreateReshapeLayer(imgdnn_tensor input_tensor,
                                   unsigned int *shape,
                                   uint32_t shape_count,
                                   imgdnn_quant_param quant_param);
  imgdnn_tensor CreateSoftmaxLayer(imgdnn_tensor input_tensor,
                                   float beta,
                                   unsigned int axis,
                                   imgdnn_quant_param quant_param);
  imgdnn_tensor CreateScaleLayer(imgdnn_tensor input_tensor,
                                 bool with_bias,
                                 const void *const scale_data,
                                 const void *const bias_data);
  imgdnn_tensor CreateReLULayer(imgdnn_tensor input_tensor,
                                bool has_min_clamp,
                                float min_clamp,
                                bool has_max_clamp,
                                float max_clamp,
                                float negative_slope);

  // Memory
  imgdnn_memory ImportMemory(
      void *buffer,
      size_t size,
      imgdnn_import_mem_type import_mem_type = IMGDNN_IMPORT_MEM_TYPE_CPU);
  imgdnn_memory AllocateMemory(size_t size);
  void DestroyMemory(imgdnn_memory memory);
  void *LockMemory(imgdnn_memory memory, imgdnn_lock_access lock_access);
  void UnlockMemory(imgdnn_memory memory);

  // Network
  imgdnn_network_object CreateNetworkObject(unsigned int num_inputs,
                                            imgdnn_tensor *inputs,
                                            unsigned int num_outputs,
                                            imgdnn_tensor *outputs);
  void ExecuteNetworkObject(bool blocking_execute,
                            unsigned int num_events_in_wait_list,
                            const imgdnn_event event_wait_list[],
                            imgdnn_event *event);
  void GetNetworkObjectInputs(unsigned int max_inputs,
                              imgdnn_input inputs[],
                              unsigned int *num_inputs);
  void GetNetworkObjectOutputs(unsigned int max_outputs,
                               imgdnn_output outputs[],
                               unsigned int *num_outputs);
  imgdnn_tensor_descriptor GetInputDescriptor(imgdnn_input input);
  imgdnn_tensor_descriptor GetOutputDescriptor(imgdnn_output output);
  imgdnn_tensor_descriptor GetTensorDescriptor(imgdnn_tensor tensor);
  size_t GetDescriptorSize(const imgdnn_tensor_descriptor *const descriptor);
  void AddBindingInput(imgdnn_input input, imgdnn_memory memory);
  void AddBindingOutput(imgdnn_output output, imgdnn_memory memory);

 private:
  imgdnn_device device_;
  imgdnn_network network_{nullptr};
  imgdnn_context context_{nullptr};
  imgdnn_binding binding_{nullptr};
  imgdnn_network_object network_object_{nullptr};
  std::vector<std::unique_ptr<uint8_t[]>> buffers_;
  imgdnn_tensor ConvertQuantTensorType(imgdnn_tensor tensor,
                                       imgdnn_quant_param *quant_param);
  bool CheckConfigFileExists(const std::string &hw_config_path,
                             const std::string &map_config_path);
};

}  // namespace imagination_nna
}  // namespace nnadapter
