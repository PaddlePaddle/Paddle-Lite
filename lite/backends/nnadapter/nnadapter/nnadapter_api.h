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

#include "nnadapter_micros.h"  // NOLINT
#include "nnadapter_types.h"   // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

/**
  * Get the count of avaiable devices or acquire the specified device and create
 * a context for model building
 */
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
int32_t NNAapdterModel_createFromNetwork(NNAdapterNetwork* network,
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

#ifdef __cplusplus
}
#endif
