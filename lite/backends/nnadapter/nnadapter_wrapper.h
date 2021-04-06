/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "lite/backends/nnadapter/nnadapter/nnadapter_types.h"

namespace paddle {
namespace lite {

class NNAdapter final {
 public:
  static NNAdapter& Global();
  using NNAdapterDevice_acquire_Type = int32_t (*)(const char*,
                                                   NNAdapterDevice**);
  using NNAdapterDevice_release_Type = void (*)(NNAdapterDevice* device);
  using NNAdapterDevice_getName_Type =
      int32_t (*)(const NNAdapterDevice* device, const char** name);
  using NNAdapterDevice_getVendor_Type =
      int32_t (*)(const NNAdapterDevice* device, const char** vendor);
  using NNAdapterDevice_getType_Type =
      int32_t (*)(const NNAdapterDevice* device, NNAdapterDeviceType* type);
  using NNAdapterDevice_getVersion_Type =
      int32_t (*)(const NNAdapterDevice* device, int32_t* version);

  using NNAdapterNetwork_create_Type = int32_t (*)(NNAdapterNetwork** network);
  using NNAdapterNetwork_free_Type = void (*)(NNAdapterNetwork* network);
  using NNAdapterNetwork_addOperand_Type =
      int32_t (*)(NNAdapterNetwork* network,
                  const NNAdapterOperandType* type,
                  NNAdapterOperand** operand);
  using NNAdapterNetwork_setOperand_Type =
      int32_t (*)(NNAdapterOperand* operand, const void* buffer, size_t length);
  using NNAdapterNetwork_addOperation_Type =
      int32_t (*)(NNAdapterNetwork* network,
                  NNAdapterOperationType type,
                  NNAdapterOperation** operation);
  using NNAdapterNetwork_setOperation_Type =
      int32_t (*)(NNAdapterOperation* operation,
                  uint32_t inputCount,
                  const NNAdapterOperand* inputs,
                  uint32_t outputCount,
                  const NNAdapterOperand* outputs);
  using NNAdapterNetwork_identifyInputsAndOutputs_Type =
      int32_t (*)(NNAdapterNetwork* network,
                  uint32_t inputCount,
                  const NNAdapterOperand* inputs,
                  uint32_t outputCount,
                  const NNAdapterOperand* outputs);

  using NNAapdterModel_createFromCache_Type =
      int32_t (*)(void* buffer, const size_t size, NNAdapterModel** model);
  using NNAapdterModel_createFromNetwork_Type =
      int32_t (*)(NNAdapterNetwork* network,
                  const NNAdapterDevice* const* devices,
                  uint32_t numDevices,
                  NNAdapterModel** model);
  using NNAapdterModel_free_Type = void (*)(NNAdapterModel* model);
  using NNAapdterModel_setCacheMode_Type = int32_t (*)(NNAdapterModel* model,
                                                       const char* cacheDir,
                                                       const uint8_t* token);
  using NNAdapterModel_getCacheSize_Type = int32_t (*)(NNAdapterModel* model,
                                                       size_t* size);
  using NNAdapterModel_getCacheBuffer_Type = int32_t (*)(NNAdapterModel* model,
                                                         void* buffer,
                                                         const size_t size);

  using NNAdapterExecution_create_Type =
      int32_t (*)(NNAdapterModel* model, NNAdapterExecution** execution);
  using NNAdapterExecution_free_Type = void (*)(NNAdapterExecution* execution);
  using NNAdapterExecution_setInput_Type =
      int32_t (*)(NNAdapterExecution* execution,
                  int32_t index,
                  const NNAdapterOperandType* type,
                  const void* buffer,
                  size_t length);
  using NNAdapterExecution_setOutput_Type =
      int32_t (*)(NNAdapterExecution* execution,
                  int32_t index,
                  const NNAdapterOperandType* type,
                  void* buffer,
                  size_t length);
  using NNAdapterExecution_startCompute_Type =
      int32_t (*)(NNAdapterExecution* execution);

  int32_t NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device);
  void NNAdapterDevice_release(NNAdapterDevice* device);
  int32_t NNAdapterDevice_getName(const NNAdapterDevice* device,
                                  const char** name);
  int32_t NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                                    const char** vendor);
  int32_t NNAdapterDevice_getType(const NNAdapterDevice* device,
                                  NNAdapterDeviceType* type);
  int32_t NNAdapterDevice_getVersion(const NNAdapterDevice* device,
                                     int32_t* version);

  int32_t NNAdapterNetwork_create(NNAdapterNetwork** network);
  void NNAdapterNetwork_free(NNAdapterNetwork* network);
  int32_t NNAdapterNetwork_addOperand(NNAdapterNetwork* network,
                                      const NNAdapterOperandType* type,
                                      NNAdapterOperand** operand);
  int32_t NNAdapterNetwork_setOperand(NNAdapterOperand* operand,
                                      const void* buffer,
                                      size_t length);
  int32_t NNAdapterNetwork_addOperation(NNAdapterNetwork* network,
                                        NNAdapterOperationType type,
                                        NNAdapterOperation** operation);
  int32_t NNAdapterNetwork_setOperation(NNAdapterOperation* operation,
                                        uint32_t inputCount,
                                        const NNAdapterOperand* inputs,
                                        uint32_t outputCount,
                                        const NNAdapterOperand* outputs);
  int32_t NNAdapterNetwork_identifyInputsAndOutputs(
      NNAdapterNetwork* network,
      uint32_t inputCount,
      const NNAdapterOperand* inputs,
      uint32_t outputCount,
      const NNAdapterOperand* outputs);
  int32_t NNAapdterModel_createFromCache(void* buffer,
                                         const size_t size,
                                         NNAdapterModel** model);

  int32_t NNAapdterModel_createFromNetwork(
      NNAdapterNetwork* network,
      const NNAdapterDevice* const* devices,
      uint32_t numDevices,
      NNAdapterModel** model);
  void NNAapdterModel_free(NNAdapterModel* model);
  int32_t NNAapdterModel_setCacheMode(NNAdapterModel* model,
                                      const char* cacheDir,
                                      const uint8_t* token);
  int32_t NNAdapterModel_getCacheSize(NNAdapterModel* model, size_t* size);
  int32_t NNAdapterModel_getCacheBuffer(NNAdapterModel* model,
                                        void* buffer,
                                        const size_t size);

  int32_t NNAdapterExecution_create(NNAdapterModel* model,
                                    NNAdapterExecution** execution);
  void NNAdapterExecution_free(NNAdapterExecution* execution);
  int32_t NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                      int32_t index,
                                      const NNAdapterOperandType* type,
                                      const void* buffer,
                                      size_t length);
  int32_t NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                       int32_t index,
                                       const NNAdapterOperandType* type,
                                       void* buffer,
                                       size_t length);
  int32_t NNAdapterExecution_startCompute(NNAdapterExecution* execution);

 private:
  NNAdapter();
  NNAdapter(const NNAdapter&) = delete;
  NNAdapter& operator=(const NNAdapter&) = delete;
  bool Init();
  void* library_{nullptr};

  NNAdapterDevice_acquire_Type NNAdapterDevice_acquire_{nullptr};
  NNAdapterDevice_release_Type NNAdapterDevice_release_{nullptr};
  NNAdapterDevice_getName_Type NNAdapterDevice_getName_{nullptr};
  NNAdapterDevice_getVendor_Type NNAdapterDevice_getVendor_{nullptr};
  NNAdapterDevice_getType_Type NNAdapterDevice_getType_{nullptr};
  NNAdapterDevice_getVersion_Type NNAdapterDevice_getVersion_{nullptr};

  NNAdapterNetwork_create_Type NNAdapterNetwork_create_{nullptr};
  NNAdapterNetwork_free_Type NNAdapterNetwork_free_{nullptr};
  NNAdapterNetwork_addOperand_Type NNAdapterNetwork_addOperand_{nullptr};
  NNAdapterNetwork_setOperand_Type NNAdapterNetwork_setOperand_{nullptr};
  NNAdapterNetwork_addOperation_Type NNAdapterNetwork_addOperation_{nullptr};
  NNAdapterNetwork_setOperation_Type NNAdapterNetwork_setOperation_{nullptr};
  NNAdapterNetwork_identifyInputsAndOutputs_Type
      NNAdapterNetwork_identifyInputsAndOutputs_{nullptr};

  NNAapdterModel_createFromCache_Type NNAapdterModel_createFromCache_{nullptr};
  NNAapdterModel_createFromNetwork_Type NNAapdterModel_createFromNetwork_{
      nullptr};
  NNAapdterModel_free_Type NNAapdterModel_free_{nullptr};
  NNAapdterModel_setCacheMode_Type NNAapdterModel_setCacheMode_{nullptr};
  NNAdapterModel_getCacheSize_Type NNAdapterModel_getCacheSize_{nullptr};
  NNAdapterModel_getCacheBuffer_Type NNAdapterModel_getCacheBuffer_{nullptr};

  NNAdapterExecution_create_Type NNAdapterExecution_create_{nullptr};
  NNAdapterExecution_free_Type NNAdapterExecution_free_{nullptr};
  NNAdapterExecution_setInput_Type NNAdapterExecution_setInput_{nullptr};
  NNAdapterExecution_setOutput_Type NNAdapterExecution_setOutput_{nullptr};
  NNAdapterExecution_startCompute_Type NNAdapterExecution_startCompute_{
      nullptr};
};
}  // namespace lite
}  // namespace paddle
