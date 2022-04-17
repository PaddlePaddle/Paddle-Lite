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

#include "lite/backends/nnadapter/nnadapter/include/nnadapter/nnadapter.h"

namespace paddle {
namespace lite {

class NNAdapterWrapper final {
 public:
  static NNAdapterWrapper& Global();
  bool Supported() { return initialized_ && supported_; }

  typedef int (*NNAdapter_getVersion_fn)(uint32_t* version);
  typedef int (*NNAdapter_getDeviceCount_fn)(uint32_t* numDevices);
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
  typedef int (*NNAdapterContext_create_fn)(NNAdapterDevice** devices,
                                            uint32_t num_devices,
                                            const char* properties,
                                            int (*callback)(int event_id,
                                                            void* user_data),
                                            NNAdapterContext** context);
  typedef void (*NNAdapterContext_destroy_fn)(NNAdapterContext* context);
  typedef int (*NNAdapterModel_create_fn)(NNAdapterModel** model);
  typedef void (*NNAdapterModel_destroy_fn)(NNAdapterModel* model);
  typedef int (*NNAdapterModel_finish_fn)(NNAdapterModel* model);
  typedef int (*NNAdapterModel_addOperand_fn)(NNAdapterModel* model,
                                              const NNAdapterOperandType* type,
                                              NNAdapterOperand** operand);
  typedef int (*NNAdapterModel_setOperandValue_fn)(NNAdapterOperand* operand,
                                                   void* buffer,
                                                   uint32_t length,
                                                   bool copy);
  typedef int (*NNAdapterModel_getOperandType_fn)(NNAdapterOperand* operand,
                                                  NNAdapterOperandType** type);
  typedef int (*NNAdapterModel_addOperation_fn)(
      NNAdapterModel* model,
      NNAdapterOperationType type,
      uint32_t input_count,
      NNAdapterOperand** input_operands,
      uint32_t output_count,
      NNAdapterOperand** output_operands,
      NNAdapterOperation** operation);
  typedef int (*NNAdapterModel_identifyInputsAndOutputs_fn)(
      NNAdapterModel* model,
      uint32_t input_count,
      NNAdapterOperand** input_operands,
      uint32_t output_count,
      NNAdapterOperand** output_operands);
  typedef int (*NNAdapterModel_getSupportedOperations_fn)(
      const NNAdapterModel* model,
      NNAdapterContext* context,
      bool* supported_operations);
  typedef int (*NNAdapterCompilation_create_fn)(
      NNAdapterModel* model,
      const char* cache_token,
      void* cache_buffer,
      uint32_t cache_length,
      const char* cache_dir,
      NNAdapterContext* context,
      NNAdapterCompilation** compilation);
  typedef void (*NNAdapterCompilation_destroy_fn)(
      NNAdapterCompilation* compilation);
  typedef int (*NNAdapterCompilation_finish_fn)(
      NNAdapterCompilation* compilation);
  typedef int (*NNAdapterCompilation_queryInputsAndOutputs_fn)(
      NNAdapterCompilation* compilation,
      uint32_t* input_count,
      NNAdapterOperandType** input_types,
      uint32_t* output_count,
      NNAdapterOperandType** output_types);
  typedef int (*NNAdapterExecution_create_fn)(NNAdapterCompilation* compilation,
                                              NNAdapterExecution** execution);
  typedef void (*NNAdapterExecution_destroy_fn)(NNAdapterExecution* execution);
  typedef int (*NNAdapterExecution_setInput_fn)(
      NNAdapterExecution* execution,
      int32_t index,
      void* memory,
      void* (*access)(void* memory,
                      NNAdapterOperandType* type,
                      void* device_buffer));
  typedef int (*NNAdapterExecution_setOutput_fn)(
      NNAdapterExecution* execution,
      int32_t index,
      void* memory,
      void* (*access)(void* memory,
                      NNAdapterOperandType* type,
                      void* device_buffer));
  typedef int (*NNAdapterExecution_compute_fn)(NNAdapterExecution* execution);

#define NNADAPTER_DECLARE_FUNCTION(name) name##_fn name;

  NNADAPTER_DECLARE_FUNCTION(NNAdapter_getVersion)
  NNADAPTER_DECLARE_FUNCTION(NNAdapter_getDeviceCount)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_acquire)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_release)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getName)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVendor)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getType)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVersion)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterContext_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterContext_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_finish)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_addOperand)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_setOperandValue)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_getOperandType)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_addOperation)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_identifyInputsAndOutputs)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_getSupportedOperations)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_finish)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterCompilation_queryInputsAndOutputs)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_create)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_destroy)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setInput)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setOutput)
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_compute)
#undef NNADAPTER_DECLARE_FUNCTION

 private:
  NNAdapterWrapper();
  NNAdapterWrapper(const NNAdapterWrapper&) = delete;
  NNAdapterWrapper& operator=(const NNAdapterWrapper&) = delete;
  ~NNAdapterWrapper();
  bool Initialize();
  bool initialized_{false};
  bool supported_{false};
  void* library_{nullptr};
};

inline int NNAdapter_getVersion_invoke(uint32_t* version) {
  return NNAdapterWrapper::Global().NNAdapter_getVersion(version);
}

inline int NNAdapter_getDeviceCount_invoke(uint32_t* numDevices) {
  return NNAdapterWrapper::Global().NNAdapter_getDeviceCount(numDevices);
}

inline int NNAdapterDevice_acquire_invoke(const char* name,
                                          NNAdapterDevice** device) {
  return NNAdapterWrapper::Global().NNAdapterDevice_acquire(name, device);
}

inline void NNAdapterDevice_release_invoke(NNAdapterDevice* device) {
  NNAdapterWrapper::Global().NNAdapterDevice_release(device);
}

inline int NNAdapterDevice_getName_invoke(const NNAdapterDevice* device,
                                          const char** name) {
  return NNAdapterWrapper::Global().NNAdapterDevice_getName(device, name);
}

inline int NNAdapterDevice_getVendor_invoke(const NNAdapterDevice* device,
                                            const char** vendor) {
  return NNAdapterWrapper::Global().NNAdapterDevice_getVendor(device, vendor);
}

inline int NNAdapterDevice_getType_invoke(const NNAdapterDevice* device,
                                          NNAdapterDeviceType* type) {
  return NNAdapterWrapper::Global().NNAdapterDevice_getType(device, type);
}

inline int NNAdapterDevice_getVersion_invoke(const NNAdapterDevice* device,
                                             int32_t* version) {
  return NNAdapterWrapper::Global().NNAdapterDevice_getVersion(device, version);
}

inline int NNAdapterContext_create_invoke(NNAdapterDevice** devices,
                                          uint32_t num_devices,
                                          const char* properties,
                                          int (*callback)(int event_id,
                                                          void* user_data),
                                          NNAdapterContext** context) {
  return NNAdapterWrapper::Global().NNAdapterContext_create(
      devices, num_devices, properties, callback, context);
}

inline void NNAdapterContext_destroy_invoke(NNAdapterContext* context) {
  NNAdapterWrapper::Global().NNAdapterContext_destroy(context);
}

inline int NNAdapterModel_create_invoke(NNAdapterModel** model) {
  return NNAdapterWrapper::Global().NNAdapterModel_create(model);
}

inline void NNAdapterModel_destroy_invoke(NNAdapterModel* model) {
  NNAdapterWrapper::Global().NNAdapterModel_destroy(model);
}

inline int NNAdapterModel_finish_invoke(NNAdapterModel* model) {
  return NNAdapterWrapper::Global().NNAdapterModel_finish(model);
}

inline int NNAdapterModel_addOperand_invoke(NNAdapterModel* model,
                                            const NNAdapterOperandType* type,
                                            NNAdapterOperand** operand) {
  return NNAdapterWrapper::Global().NNAdapterModel_addOperand(
      model, type, operand);
}

inline int NNAdapterModel_setOperandValue_invoke(NNAdapterOperand* operand,
                                                 void* buffer,
                                                 uint32_t length,
                                                 bool copy) {
  return NNAdapterWrapper::Global().NNAdapterModel_setOperandValue(
      operand, buffer, length, copy);
}

inline int NNAdapterModel_getOperandType_invoke(NNAdapterOperand* operand,
                                                NNAdapterOperandType** type) {
  return NNAdapterWrapper::Global().NNAdapterModel_getOperandType(operand,
                                                                  type);
}

inline int NNAdapterModel_addOperation_invoke(
    NNAdapterModel* model,
    NNAdapterOperationType type,
    uint32_t input_count,
    NNAdapterOperand** input_operands,
    uint32_t output_count,
    NNAdapterOperand** output_operands,
    NNAdapterOperation** operation) {
  return NNAdapterWrapper::Global().NNAdapterModel_addOperation(model,
                                                                type,
                                                                input_count,
                                                                input_operands,
                                                                output_count,
                                                                output_operands,
                                                                operation);
}

inline int NNAdapterModel_identifyInputsAndOutputs_invoke(
    NNAdapterModel* model,
    uint32_t input_count,
    NNAdapterOperand** input_operands,
    uint32_t output_count,
    NNAdapterOperand** output_operands) {
  return NNAdapterWrapper::Global().NNAdapterModel_identifyInputsAndOutputs(
      model, input_count, input_operands, output_count, output_operands);
}

inline int NNAdapterModel_getSupportedOperations_invoke(
    const NNAdapterModel* model,
    NNAdapterContext* context,
    bool* supported_operations) {
  return NNAdapterWrapper::Global().NNAdapterModel_getSupportedOperations(
      model, context, supported_operations);
}

inline int NNAdapterCompilation_create_invoke(
    NNAdapterModel* model,
    const char* cache_token,
    void* cache_buffer,
    uint32_t cache_length,
    const char* cache_dir,
    NNAdapterContext* context,
    NNAdapterCompilation** compilation) {
  return NNAdapterWrapper::Global().NNAdapterCompilation_create(model,
                                                                cache_token,
                                                                cache_buffer,
                                                                cache_length,
                                                                cache_dir,
                                                                context,
                                                                compilation);
}

inline void NNAdapterCompilation_destroy_invoke(
    NNAdapterCompilation* compilation) {
  NNAdapterWrapper::Global().NNAdapterCompilation_destroy(compilation);
}

inline int NNAdapterCompilation_finish_invoke(
    NNAdapterCompilation* compilation) {
  return NNAdapterWrapper::Global().NNAdapterCompilation_finish(compilation);
}

inline int NNAdapterCompilation_queryInputsAndOutputs_invoke(
    NNAdapterCompilation* compilation,
    uint32_t* input_count,
    NNAdapterOperandType** input_types,
    uint32_t* output_count,
    NNAdapterOperandType** output_types) {
  return NNAdapterWrapper::Global().NNAdapterCompilation_queryInputsAndOutputs(
      compilation, input_count, input_types, output_count, output_types);
}

inline int NNAdapterExecution_create_invoke(NNAdapterCompilation* compilation,
                                            NNAdapterExecution** execution) {
  return NNAdapterWrapper::Global().NNAdapterExecution_create(compilation,
                                                              execution);
}

inline void NNAdapterExecution_destroy_invoke(NNAdapterExecution* execution) {
  NNAdapterWrapper::Global().NNAdapterExecution_destroy(execution);
}

inline int NNAdapterExecution_setInput_invoke(
    NNAdapterExecution* execution,
    int32_t index,
    void* memory,
    void* (*access)(void* memory,
                    NNAdapterOperandType* type,
                    void* device_buffer)) {
  return NNAdapterWrapper::Global().NNAdapterExecution_setInput(
      execution, index, memory, access);
}

inline int NNAdapterExecution_setOutput_invoke(
    NNAdapterExecution* execution,
    int32_t index,
    void* memory,
    void* (*access)(void* memory,
                    NNAdapterOperandType* type,
                    void* device_buffer)) {
  return NNAdapterWrapper::Global().NNAdapterExecution_setOutput(
      execution, index, memory, access);
}

inline int NNAdapterExecution_compute_invoke(NNAdapterExecution* execution) {
  return NNAdapterWrapper::Global().NNAdapterExecution_compute(execution);
}

}  // namespace lite
}  // namespace paddle
