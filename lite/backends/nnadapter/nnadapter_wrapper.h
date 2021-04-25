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
  bool Supported() { return initialized_ && supported_; }

  typedef int (*NNAdapterDevice_acquire_fn)(const char* name,
                                            NNAdapterDevice** device);
  typedef void (*NNAdapterDevice_release_fn)(NNAdapterDevice* device);
  typedef int (*NNAdapterDevice_getName_fn)(const NNAdapterDevice* device,
                                            const char** name);
  typedef int (*NNAdapterDevice_getVendor_fn)(const NNAdapterDevice* device,
                                              const char** vendor);
  typedef int (*NNAdapterDevice_getType_fn)(const NNAdapterDevice* device,
                                            NNAdapterDeviceType* type);
  typedef int (*NNAdapterDevice_getVersion_fn)(const NNAdapterDevice* device,
                                               int32_t* version);

  typedef int (*NNAdapterModel_create_fn)(NNAdapterModel** model);
  typedef void (*NNAdapterModel_destroy_fn)(NNAdapterModel* model);
  typedef int (*NNAdapterModel_finish_fn)(NNAdapterModel* model);
  typedef int (*NNAdapterModel_addOperand_fn)(NNAdapterModel* model,
                                              const NNAdapterOperandType* type,
                                              NNAdapterOperand** operand);
  typedef int (*NNAdapterModel_setOperand_fn)(NNAdapterOperand* operand,
                                              void* buffer,
                                              size_t length);
  typedef int (*NNAdapterModel_addOperation_fn)(NNAdapterModel* model,
                                                NNAdapterOperationType type,
                                                NNAdapterOperation** operation);
  typedef int (*NNAdapterModel_setOperation_fn)(NNAdapterOperation* operation,
                                                uint32_t input_count,
                                                NNAdapterOperand** inputs,
                                                uint32_t output_count,
                                                NNAdapterOperand** outputs);
  typedef int (*NNAdapterModel_identifyInputsAndOutputs_fn)(
      NNAdapterModel* model,
      uint32_t input_count,
      NNAdapterOperand** inputs,
      uint32_t output_count,
      NNAdapterOperand** outputs);

  typedef int (*NNAdapterCompilation_create_fn)(
      NNAdapterModel* model,
      const char* cache_key,
      void* cache_buffer,
      size_t cache_length,
      const char* cache_dir,
      NNAdapterDevice** devices,
      uint32_t num_devices,
      NNAdapterCompilation** compilation);
  typedef void (*NNAdapterCompilation_destroy_fn)(
      NNAdapterCompilation* compilation);
  typedef int (*NNAdapterCompilation_finish_fn)(
      NNAdapterCompilation* compilation);

  typedef int (*NNAdapterExecution_create_fn)(NNAdapterCompilation* compilation,
                                              NNAdapterExecution** execution);
  typedef void (*NNAdapterExecution_destroy_fn)(NNAdapterExecution* execution);
  typedef int (*NNAdapterExecution_setInput_fn)(NNAdapterExecution* execution,
                                                int32_t index,
                                                const uint32_t* dimensions,
                                                uint32_t dimension_count,
                                                void* buffer,
                                                size_t length);
  typedef int (*NNAdapterExecution_setOutput_fn)(NNAdapterExecution* execution,
                                                 int32_t index,
                                                 const uint32_t* dimensions,
                                                 uint32_t dimension_count,
                                                 void* buffer,
                                                 size_t length);
  typedef int (*NNAdapterExecution_compute_fn)(NNAdapterExecution* execution);

#define NNADAPTER_DECLARE_FUNCTION(name) name##_fn name;

  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_acquire)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_release)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getName)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVendor)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getType)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVersion)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_finish)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_addOperand)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_setOperand)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_addOperation)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_setOperation)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_identifyInputsAndOutputs)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_finish)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setInput)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setOutput)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_compute)
#undef NNADAPTER_DECLARE_FUNCTION

 private:
  NNAdapter();
  NNAdapter(const NNAdapter&) = delete;
  NNAdapter& operator=(const NNAdapter&) = delete;
  bool Initialize();
  bool initialized_{false};
  bool supported_{false};
  void* library_{nullptr};
};

inline int NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device) {
  return NNAdapter::Global().NNAdapterDevice_acquire(name, device);
}

inline void NNAdapterDevice_release(NNAdapterDevice* device) {
  NNAdapter::Global().NNAdapterDevice_release(device);
}

inline int NNAdapterDevice_getName(const NNAdapterDevice* device,
                                   const char** name) {
  return NNAdapter::Global().NNAdapterDevice_getName(device, name);
}

inline int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                                     const char** vendor) {
  return NNAdapter::Global().NNAdapterDevice_getVendor(device, vendor);
}

inline int NNAdapterDevice_getType(const NNAdapterDevice* device,
                                   NNAdapterDeviceType* type) {
  return NNAdapter::Global().NNAdapterDevice_getType(device, type);
}

inline int NNAdapterDevice_getVersion(const NNAdapterDevice* device,
                                      int32_t* version) {
  return NNAdapter::Global().NNAdapterDevice_getVersion(device, version);
}

inline int NNAdapterModel_create(NNAdapterModel** model) {
  return NNAdapter::Global().NNAdapterModel_create(model);
}

inline void NNAdapterModel_destroy(NNAdapterModel* model) {
  NNAdapter::Global().NNAdapterModel_destroy(model);
}

inline int NNAdapterModel_finish(NNAdapterModel* model) {
  return NNAdapter::Global().NNAdapterModel_finish(model);
}

inline int NNAdapterModel_addOperand(NNAdapterModel* model,
                                     const NNAdapterOperandType* type,
                                     NNAdapterOperand** operand) {
  return NNAdapter::Global().NNAdapterModel_addOperand(model, type, operand);
}

inline int NNAdapterModel_setOperand(NNAdapterOperand* operand,
                                     void* buffer,
                                     size_t length) {
  return NNAdapter::Global().NNAdapterModel_setOperand(operand, buffer, length);
}

inline int NNAdapterModel_addOperation(NNAdapterModel* model,
                                       NNAdapterOperationType type,
                                       NNAdapterOperation** operation) {
  return NNAdapter::Global().NNAdapterModel_addOperation(
      model, type, operation);
}

inline int NNAdapterModel_setOperation(NNAdapterOperation* operation,
                                       uint32_t input_count,
                                       NNAdapterOperand** inputs,
                                       uint32_t output_count,
                                       NNAdapterOperand** outputs) {
  return NNAdapter::Global().NNAdapterModel_setOperation(
      operation, input_count, inputs, output_count, outputs);
}

inline int NNAdapterModel_identifyInputsAndOutputs(NNAdapterModel* model,
                                                   uint32_t input_count,
                                                   NNAdapterOperand** inputs,
                                                   uint32_t output_count,
                                                   NNAdapterOperand** outputs) {
  return NNAdapter::Global().NNAdapterModel_identifyInputsAndOutputs(
      model, input_count, inputs, output_count, outputs);
}

inline int NNAdapterCompilation_create(NNAdapterModel* model,
                                       const char* cache_key,
                                       void* cache_buffer,
                                       size_t cache_length,
                                       const char* cache_dir,
                                       NNAdapterDevice** devices,
                                       uint32_t num_devices,
                                       NNAdapterCompilation** compilation) {
  return NNAdapter::Global().NNAdapterCompilation_create(model,
                                                         cache_key,
                                                         cache_buffer,
                                                         cache_length,
                                                         cache_dir,
                                                         devices,
                                                         num_devices,
                                                         compilation);
}

inline void NNAdapterCompilation_destroy(NNAdapterCompilation* compilation) {
  NNAdapter::Global().NNAdapterCompilation_destroy(compilation);
}

inline int NNAdapterCompilation_finish(NNAdapterCompilation* compilation) {
  return NNAdapter::Global().NNAdapterCompilation_finish(compilation);
}

inline int NNAdapterExecution_create(NNAdapterCompilation* compilation,
                                     NNAdapterExecution** execution) {
  return NNAdapter::Global().NNAdapterExecution_create(compilation, execution);
}

inline void NNAdapterExecution_destroy(NNAdapterExecution* execution) {
  NNAdapter::Global().NNAdapterExecution_destroy(execution);
}

inline int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                       int32_t index,
                                       const uint32_t* dimensions,
                                       uint32_t dimension_count,
                                       void* buffer,
                                       size_t length) {
  return NNAdapter::Global().NNAdapterExecution_setInput(
      execution, index, dimensions, dimension_count, buffer, length);
}

inline int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                        int32_t index,
                                        const uint32_t* dimensions,
                                        uint32_t dimension_count,
                                        void* buffer,
                                        size_t length) {
  return NNAdapter::Global().NNAdapterExecution_setOutput(
      execution, index, dimensions, dimension_count, buffer, length);
}

inline int NNAdapterExecution_compute(NNAdapterExecution* execution) {
  return NNAdapter::Global().NNAdapterExecution_compute(execution);
}

}  // namespace lite
}  // namespace paddle
