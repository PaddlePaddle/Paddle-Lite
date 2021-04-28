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
/**
 * Get the name of the specified device.
 */
int NNAdapterDevice_getName(const NNAdapterDevice* device, const char** name);
/**
 * Get the vendor of the specified device.
 */
int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                              const char** vendor);
/**
 * Get the type of the specified device.
 * The supported types are listed in NNAdapterDeviceCode.
 */
int NNAdapterDevice_getType(const NNAdapterDevice* device,
                            NNAdapterDeviceType* type);
/**
 * Get the driver version of the specified device.
 */
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
/**
 * Indicate that we have finished building a model, it must only called once.
 */
int NNAdapterModel_finish(NNAdapterModel* model);
/**
 * Add an operand to a model.
 */
int NNAdapterModel_addOperand(NNAdapterModel* model,
                              const NNAdapterOperandType* type,
                              NNAdapterOperand** operand);
/**
 * Set the memory for an constant operand.
 */
int NNAdapterModel_setOperand(NNAdapterOperand* operand,
                              void* buffer,
                              uint32_t length);
/**
 * Add an operation to a model.
 */
int NNAdapterModel_addOperation(NNAdapterModel* model,
                                NNAdapterOperationType type,
                                NNAdapterOperation** operation);
/**
 * Set the input and output operands of the specified operation.
 */
int NNAdapterModel_setOperation(NNAdapterOperation* operation,
                                uint32_t input_count,
                                NNAdapterOperand** input_operands,
                                uint32_t output_count,
                                NNAdapterOperand** output_operands);
/**
 * Indentify the input and output operands of the specified model.
 */
int NNAdapterModel_identifyInputsAndOutputs(NNAdapterModel* model,
                                            uint32_t input_count,
                                            NNAdapterOperand** input_operands,
                                            uint32_t output_count,
                                            NNAdapterOperand** output_operands);

/**
 * Compile the model to the hardware-related binary program or load the cached
 * binary program from memory or file system.
 * If cache_key, cache_buffer and cache_length is specified, load the binary
 * program from memory directly.
 * If cache_key and cache_dir is specified, find and load the cached binary
 * program from the cache files directly.
 * If no cache parameter is specified or the cache files are not found, then
 * compile the given model to the binary program of target devices.
 */
int NNAdapterCompilation_create(NNAdapterModel* model,
                                const char* cache_key,
                                void* cache_buffer,
                                uint32_t cache_length,
                                const char* cache_dir,
                                NNAdapterDevice** devices,
                                uint32_t num_devices,
                                NNAdapterCompilation** compilation);
/**
 * Destroy the hardware-related binary program.
 */
void NNAdapterCompilation_destroy(NNAdapterCompilation* compilation);
/**
 * Indicate that we have finished building a compilation and start to compile
 * the
 * model to the hardware-releate binary program, it must only called once.
 */
int NNAdapterCompilation_finish(NNAdapterCompilation* compilation);
/**
 * Query the information of input and output operands of the hardware-releate
 * binary program, it must called after `NNAdapterCompilation_finish`. The
 * function should be called twice: firstly, only set input_count and
 * output_count to obtain the count of inputs and outputs, secondly, allocate
 * memory for the pointers of input and output types, then call it again.
 */
int NNAdapterCompilation_queryInputsAndOutputs(
    NNAdapterCompilation* compilation,
    uint32_t* input_count,
    NNAdapterOperandType** input_types,
    uint32_t* output_count,
    NNAdapterOperandType** output_types);

/**
 * Create an execution plan to execute the hardware-related binary program.
 */
int NNAdapterExecution_create(NNAdapterCompilation* compilation,
                              NNAdapterExecution** execution);
/**
 * Destroy an execution plan.
 */
void NNAdapterExecution_destroy(NNAdapterExecution* execution);
/**
 * Set the real dimensions and buffer of the model inputs.
 */
int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                int32_t index,
                                const int32_t* dimensions,
                                uint32_t dimension_count,
                                void* buffer,
                                uint32_t length);
/**
 * Set the real dimensions and buffer of the model outputs.
 */
int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                 int32_t index,
                                 const int32_t* dimensions,
                                 uint32_t dimension_count,
                                 void* buffer,
                                 uint32_t length);
/**
 * Start to run the execution synchronously.
 */
int NNAdapterExecution_compute(NNAdapterExecution* execution);

#ifdef __cplusplus
}
#endif
