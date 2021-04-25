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
 * Acquire the specified device with device name and create a context for model
 * compilation.
 */
int NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device);
/**
 * Release the target device and its context.
 */
void NNAdapterDevice_release(NNAdapterDevice* device);
int NNAdapterDevice_getName(const NNAdapterDevice* device, const char** name);
int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                              const char** vendor);
int NNAdapterDevice_getType(const NNAdapterDevice* device,
                            NNAdapterDeviceType* type);
int NNAdapterDevice_getVersion(const NNAdapterDevice* device, int32_t* version);

/**
 * Create a hardware-independent neural networks model.
 */
int NNAdapterModel_create(NNAdapterModel** model);
/**
 * Destroy the model that free all of resources of the model includes the memory
 * of constant operands, quantization parameters, etc.
 */
void NNAdapterModel_destroy(NNAdapterModel* model);
int NNAdapterModel_finish(NNAdapterModel* model);
int NNAdapterModel_addOperand(NNAdapterModel* model,
                              const NNAdapterOperandType* type,
                              NNAdapterOperand** operand);
int NNAdapterModel_setOperand(NNAdapterOperand* operand,
                              void* buffer,
                              size_t length);
int NNAdapterModel_addOperation(NNAdapterModel* model,
                                NNAdapterOperationType type,
                                NNAdapterOperation** operation);
int NNAdapterModel_setOperation(NNAdapterOperation* operation,
                                uint32_t input_count,
                                NNAdapterOperand** inputs,
                                uint32_t output_count,
                                NNAdapterOperand** outputs);
int NNAdapterModel_identifyInputsAndOutputs(NNAdapterModel* model,
                                            uint32_t input_count,
                                            NNAdapterOperand** inputs,
                                            uint32_t output_count,
                                            NNAdapterOperand** outputs);

/**
 * Compile the model to the hardware-related binary program or load the cached
 * binary program from
 * memory or file system.
 * If cache_key, cache_buffer and cache_length is specified, load the binary
 * program from memory
 * directly.
 * If cache_key and cache_dir is specified, find and load the cached binary
 * program from the cache
 * files directly.
 * If no cache parameter is specified or the cache files are not found, then
 * compile the given model
 * to the binary program of target devices.
 */
int NNAdapterCompilation_create(NNAdapterModel* model,
                                const char* cache_key,
                                void* cache_buffer,
                                size_t cache_length,
                                const char* cache_dir,
                                NNAdapterDevice** devices,
                                uint32_t num_devices,
                                NNAdapterCompilation** compilation);
/**
 * Destroy the hardware-related binary program.
 */
void NNAdapterCompilation_destroy(NNAdapterCompilation* compilation);
int NNAdapterCompilation_finish(NNAdapterCompilation* compilation);

/**
 * Create an execution plan to execute the hardware-related binary program.
 */
int NNAdapterExecution_create(NNAdapterCompilation* compilation,
                              NNAdapterExecution** execution);
/**
 * Destroy an execution plan.
 */
void NNAdapterExecution_destroy(NNAdapterExecution* execution);
int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                int32_t index,
                                const uint32_t* dimensions,
                                uint32_t dimension_count,
                                void* buffer,
                                size_t length);
int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                 int32_t index,
                                 const uint32_t* dimensions,
                                 uint32_t dimension_count,
                                 void* buffer,
                                 size_t length);
int NNAdapterExecution_compute(NNAdapterExecution* execution);

#ifdef __cplusplus
}
#endif
