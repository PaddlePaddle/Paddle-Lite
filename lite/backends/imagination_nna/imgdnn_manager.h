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
#include "lite/utils/logging.h"

namespace paddle {
namespace lite {
namespace imagination_nna {

// Fix the compilation error 'undefined reference to
// paddle::lite::replace_stl::ostream' in tiny_publish mode
#define IMG_CHECK_SUCCESS(err_code_) \
  CHECK_EQ(static_cast<int>(err_code_), static_cast<int>(IMGDNN_SUCCESS))

class ImgdnnManager {
  imgdnn_err_code err_;
  imgdnn_device device_;
  imgdnn_network net_{nullptr};
  imgdnn_context context_{nullptr};
  imgdnn_binding binding_{nullptr};
  imgdnn_network_object net_obj_{nullptr};
  std::vector<std::unique_ptr<uint8_t[]>> coef_pool;

  imgdnn_tensor ConvertQuantTensorType(imgdnn_tensor a_tensor,
                                       imgdnn_quant_param *dst_quant_param);

  bool CheckConfigFileExists(const std::string &hwconfig,
                             const std::string &mapconfig);

 public:
  ImgdnnManager();

  virtual ~ImgdnnManager() {
    if (net_obj_) err_ = imgdnnNetworkObjectDestroy(net_obj_);
    if (context_) err_ = imgdnnContextDestroy(context_);
    if (binding_) err_ = imgdnnBindingDestroy(binding_);
    if (net_) err_ = imgdnnNetworkDestroy(net_);
  }

  uint8_t *GetBufromPool(size_t size) {
    coef_pool.emplace_back(new uint8_t[size]);
    return coef_pool.back().get();
  }

  imgdnn_network GetNetwork() { return net_; }

  imgdnn_tensor CreateInputTensor(imgdnn_tensor_descriptor *desc) {
    return imgdnnNetworkInput(net_, desc, &err_);
  }

  imgdnn_tensor CreateFixedInputTensor(imgdnn_tensor_descriptor *desc,
                                       const void *const fixed_data,
                                       bool mem_copy) {
    imgdnn_tensor fixed_input;
    if (mem_copy) {
      size_t buffer_size = imgdnnGetDescriptorSize(desc, &err_);
      void *buf = GetBufromPool(buffer_size);
      memcpy(buf, fixed_data, buffer_size);
      fixed_input = imgdnnNetworkFixedInput(net_, desc, buf, &err_);
    } else {
      fixed_input = imgdnnNetworkFixedInput(net_, desc, fixed_data, &err_);
    }
    return fixed_input;
  }

  imgdnn_tensor CreateConvolutionLayer(imgdnn_tensor input_tensor,
                                       imgdnn_tensor weights_tensor,
                                       imgdnn_tensor bias_tensor,
                                       imgdnn_quant_param dst_quant_param,
                                       unsigned int stride[2],
                                       unsigned int pad_begin[2],
                                       unsigned int pad_end[2],
                                       unsigned int dilation[2],
                                       bool use_dwconv = false);
  imgdnn_tensor CreateBatchNormLayer(imgdnn_tensor input_tensor,
                                     const void *const avg_in,
                                     const void *const var_in,
                                     const float eps);
  imgdnn_tensor CreatePoolingLayer(imgdnn_tensor in_tensor,
                                   imgdnn_quant_param dst_quant_param,
                                   const unsigned int size[2],
                                   const unsigned int stride[2],
                                   const unsigned int pad_to_begin[2],
                                   const unsigned int pad_to_end[2],
                                   imgdnn_pooling_type type);
  imgdnn_tensor CreateFullyConnectedLayer(imgdnn_tensor input_tensor,
                                          imgdnn_tensor weights_tensor,
                                          imgdnn_tensor bias_tensor,
                                          imgdnn_quant_param dst_quant_param);
  imgdnn_tensor CreateSoftmaxLayer(imgdnn_tensor in_tensor,
                                   float beta,
                                   unsigned int axis,
                                   imgdnn_quant_param dst_quant_param);
  imgdnn_tensor CreateScaleLayer(imgdnn_tensor input_tensor,
                                 bool with_biasscale,
                                 const void *const scale,
                                 const void *const bias);

  imgdnn_tensor CreateReLULayer(imgdnn_tensor in_tensor,
                                bool has_min_clamp,
                                float min_clamp,
                                bool has_max_clamp,
                                float max_clamp,
                                float negative_slope) {
    imgdnn_tensor relu_tensor = imgdnnNetworkReLUOp(net_,
                                                    in_tensor,
                                                    has_min_clamp,
                                                    min_clamp,
                                                    has_max_clamp,
                                                    max_clamp,
                                                    negative_slope,
                                                    &err_);
    IMG_CHECK_SUCCESS(err_) << "ReLU OP fails";

    imgdnn_tensor_descriptor in_desc, relu_desc;
    imgdnnGetTensorDescriptor(in_tensor, &in_desc);
    imgdnnGetTensorDescriptor(relu_tensor, &relu_desc);
    if (relu_desc.type != in_desc.type) {
      relu_tensor = imgdnnNetworkCastOp(
          net_, relu_tensor, in_desc.type, &in_desc.quant_param, &err_);
      IMG_CHECK_SUCCESS(err_) << "ReLU cast fails";
    }

    return relu_tensor;
  }

  imgdnn_network_object CreateNetworkObject(unsigned int num_inputs,
                                            imgdnn_tensor *inputs,
                                            unsigned int num_outputs,
                                            imgdnn_tensor *outputs);

  imgdnn_memory ImportMemory(
      void *memory,
      size_t size,
      imgdnn_import_mem_type import_mem_type = IMGDNN_IMPORT_MEM_TYPE_CPU) {
    imgdnn_memory mem =
        imgdnnImportMemory(context_, memory, size, import_mem_type, &err_);
    IMG_CHECK_SUCCESS(err_) << "ImportMemory fails";
    return mem;
  }

  imgdnn_memory AllocateMemory(size_t size) {
    imgdnn_memory mem = imgdnnAllocateMemory(context_, size, &err_);
    IMG_CHECK_SUCCESS(err_) << "AllocateMemory fails";
    return mem;
  }

  void DestroyMemory(imgdnn_memory memory) {
    err_ = imgdnnMemoryDestroy(memory);
    IMG_CHECK_SUCCESS(err_) << "MemoryDestroy fails";
  }

  void *LockMemory(imgdnn_memory memory, imgdnn_lock_access lock_access) {
    void *mem = imgdnnMemoryLock(memory, lock_access, &err_);
    IMG_CHECK_SUCCESS(err_) << "MemoryLock fails";
    return mem;
  }

  void UnlockMemory(imgdnn_memory memory) {
    err_ = imgdnnMemoryUnlock(memory);
    IMG_CHECK_SUCCESS(err_) << "MemoryUnLock fails";
  }

  void GetNetworkObjectInputs(unsigned int max_inputs,
                              imgdnn_input inputs[],
                              unsigned int *num_inputs) {
    CHECK(net_obj_ != nullptr) << "NetworkObject NULL when get its inputs";
    err_ =
        imgdnnNetworkObjectGetInputs(net_obj_, max_inputs, inputs, num_inputs);
    IMG_CHECK_SUCCESS(err_) << "NetworkObjectGetInputs failed!";
  }

  void GetNetworkObjectOutputs(unsigned int max_outputs,
                               imgdnn_output outputs[],
                               unsigned int *num_outputs) {
    CHECK(net_obj_ != nullptr) << "NetworkObject NULL when get its outputs";
    err_ = imgdnnNetworkObjectGetOutputs(
        net_obj_, max_outputs, outputs, num_outputs);
    IMG_CHECK_SUCCESS(err_) << "NetworkObjectGetOutputs failed!";
  }

  imgdnn_tensor_descriptor GetInputDescriptor(imgdnn_input input) {
    imgdnn_tensor_descriptor desc = imgdnnGetInputDescriptor(input, &err_);
    IMG_CHECK_SUCCESS(err_) << "GetInputDescriptors failed!";
    return desc;
  }

  imgdnn_tensor_descriptor GetOutputDescriptor(imgdnn_output output) {
    imgdnn_tensor_descriptor desc = imgdnnGetOutputDescriptor(output, &err_);
    IMG_CHECK_SUCCESS(err_) << "GetOutputDescriptors failed!";
    return desc;
  }

  imgdnn_tensor_descriptor GetTensorDescriptor(imgdnn_tensor tensor) {
    imgdnn_tensor_descriptor desc;
    err_ = imgdnnGetTensorDescriptor(tensor, &desc);
    IMG_CHECK_SUCCESS(err_) << "GetTensorDescriptors failed!";
    return desc;
  }

  size_t GetDescriptorSize(const imgdnn_tensor_descriptor *const descriptor) {
    size_t size = imgdnnGetDescriptorSize(descriptor, &err_);
    IMG_CHECK_SUCCESS(err_) << "GetDescriptorSize failed!";
    return size;
  }

  void AddBindingInput(imgdnn_input input, imgdnn_memory memory) {
    err_ = imgdnnBindingAddInput(binding_, input, memory);
    IMG_CHECK_SUCCESS(err_) << "BindingAddInput failed!";
  }

  void AddBindingOutput(imgdnn_output output, imgdnn_memory memory) {
    err_ = imgdnnBindingAddOutput(binding_, output, memory);
    IMG_CHECK_SUCCESS(err_) << "BindingAddOutput failed!";
  }

  void ExecuteNetworkObject(bool blocking_execute,
                            unsigned int num_events_in_wait_list,
                            const imgdnn_event event_wait_list[],
                            imgdnn_event *event) {
    err_ = imgdnnNetworkObjectExecute(net_obj_,
                                      binding_,
                                      blocking_execute,
                                      num_events_in_wait_list,
                                      event_wait_list,
                                      event);
    IMG_CHECK_SUCCESS(err_) << "NetworkObjectExecute failed!";
  }
};

}  // namespace imagination_nna
}  // namespace lite
}  // namespace paddle
