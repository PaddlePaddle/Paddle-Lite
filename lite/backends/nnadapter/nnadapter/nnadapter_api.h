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
int NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device);
void NNAdapterDevice_release(NNAdapterDevice* device);
int NNAdapterDevice_getName(const NNAdapterDevice* device, const char** name);
int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                              const char** vendor);
int NNAdapterDevice_getType(const NNAdapterDevice* device,
                            NNAdapterDeviceType* type);
int NNAdapterDevice_getVersion(const NNAdapterDevice* device, int32_t* version);

int NNAdapterGraph_create(NNAdapterGraph** graph);
void NNAdapterGraph_destroy(NNAdapterGraph* graph);
int NNAdapterGraph_finish(NNAdapterGraph* graph);
int NNAdapterGraph_addOperand(NNAdapterGraph* graph,
                              const NNAdapterOperandType* type,
                              NNAdapterOperand** operand);
int NNAdapterGraph_setOperand(NNAdapterOperand* operand,
                              void* buffer,
                              size_t length);
int NNAdapterGraph_addOperation(NNAdapterGraph* graph,
                                NNAdapterOperationType type,
                                NNAdapterOperation** operation);
int NNAdapterGraph_setOperation(NNAdapterOperation* operation,
                                uint32_t inputCount,
                                NNAdapterOperand** inputs,
                                uint32_t outputCount,
                                NNAdapterOperand** outputs);
int NNAdapterGraph_identifyInputsAndOutputs(NNAdapterGraph* graph,
                                            uint32_t inputCount,
                                            NNAdapterOperand** inputs,
                                            uint32_t outputCount,
                                            NNAdapterOperand** outputs);

int NNAdapterModel_createFromGraph(NNAdapterGraph* graph,
                                   NNAdapterDevice** devices,
                                   uint32_t numDevices,
                                   NNAdapterModel** model);
int NNAdapterModel_createFromCache(void* buffer,
                                   size_t length,
                                   uint32_t inputCount,
                                   const NNAdapterOperandType** inputTypes,
                                   uint32_t outputCount,
                                   const NNAdapterOperandType** outputTypes,
                                   NNAdapterDevice** devices,
                                   uint32_t numDevices,
                                   NNAdapterModel** model);
void NNAdapterModel_destroy(NNAdapterModel* model);
int NNAdapterModel_finish(NNAdapterModel* model);
int NNAdapterModel_setCaching(NNAdapterModel* model,
                              const char* cacheDir,
                              const uint8_t* token);

int NNAdapterExecution_create(NNAdapterModel* model,
                              NNAdapterExecution** execution);
void NNAdapterExecution_destroy(NNAdapterExecution* execution);
int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                int32_t index,
                                const uint32_t* dimensions,
                                uint32_t dimensionCount,
                                void* buffer,
                                size_t length);
int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                 int32_t index,
                                 const uint32_t* dimensions,
                                 uint32_t dimensionCount,
                                 void* buffer,
                                 size_t length);
int NNAdapterExecution_run(NNAdapterExecution* execution,
                           NNAdapterEvent** event);
int NNAdapterEvent_wait(NNAdapterEvent* event);
void NNAdapterEvent_destroy(NNAdapterEvent* event);

#ifdef __cplusplus
}
#endif
