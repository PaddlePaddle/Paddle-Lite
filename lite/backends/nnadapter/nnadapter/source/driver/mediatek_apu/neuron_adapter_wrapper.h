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

#include "NeuronAdapter.h"  // NOLINT

namespace nnadapter {
namespace mediatek_apu {

class NeuronAdapterWrapper final {
 public:
  static NeuronAdapterWrapper& Global();
  bool Supported() { return initialized_ && supported_; }

  typedef int (*Neuron_getVersion_fn)(uint32_t* version);
  typedef int (*NeuronModel_create_fn)(NeuronModel** model);
  typedef void (*NeuronModel_free_fn)(NeuronModel* model);
  typedef int (*NeuronModel_finish_fn)(NeuronModel* model);
  typedef int (*NeuronModel_addOperand_fn)(NeuronModel* model,
                                           const NeuronOperandType* type);
  typedef int (*NeuronModel_setOperandValue_fn)(NeuronModel* model,
                                                int32_t index,
                                                const void* buffer,
                                                size_t length);
  typedef int (*NeuronModel_setOperandSymmPerChannelQuantParams_fn)(
      NeuronModel* model,
      int32_t index,
      const NeuronSymmPerChannelQuantParams* channelQuant);
  typedef int (*NeuronModel_addOperation_fn)(NeuronModel* model,
                                             NeuronOperationType type,
                                             uint32_t inputCount,
                                             const uint32_t* inputs,
                                             uint32_t outputCount,
                                             const uint32_t* outputs);
  typedef int (*NeuronModel_addOperationExtension_fn)(
      NeuronModel* model,
      const char* name,
      const char* vendor,
      const NeuronDevice* device,
      uint32_t inputCount,
      const uint32_t* inputs,
      uint32_t outputCount,
      const uint32_t* outputs);
  typedef int (*NeuronModel_identifyInputsAndOutputs_fn)(
      NeuronModel* model,
      uint32_t inputCount,
      const uint32_t* inputs,
      uint32_t outputCount,
      const uint32_t* outputs);
  typedef int (*NeuronModel_getSupportedOperations_fn)(NeuronModel* model,
                                                       bool* supported,
                                                       uint32_t operationCount);
  typedef int (*NeuronModel_relaxComputationFloat32toFloat16_fn)(
      NeuronModel* model, bool allow);
  typedef int (*NeuronModel_restoreFromCompiledNetwork_fn)(
      NeuronModel** model,
      NeuronCompilation** compilation,
      const void* buffer,
      const size_t size);
  typedef int (*NeuronCompilation_create_fn)(NeuronModel* model,
                                             NeuronCompilation** compilation);
  typedef void (*NeuronCompilation_free_fn)(NeuronCompilation* compilation);
  typedef int (*NeuronCompilation_finish_fn)(NeuronCompilation* compilation);
  typedef int (*NeuronCompilation_setCaching_fn)(NeuronCompilation* compilation,
                                                 const char* cacheDir,
                                                 const uint8_t* token);
  typedef int (*NeuronCompilation_createForDevices_fn)(
      NeuronModel* model,
      const NeuronDevice* const* devices,
      uint32_t numDevices,
      NeuronCompilation** compilation);
  typedef int (*NeuronCompilation_getCompiledNetworkSize_fn)(
      NeuronCompilation* compilation, size_t* size);
  typedef int (*NeuronCompilation_storeCompiledNetwork_fn)(
      NeuronCompilation* compilation, void* buffer, const size_t size);
  typedef int (*NeuronExecution_create_fn)(NeuronCompilation* compilation,
                                           NeuronExecution** execution);
  typedef void (*NeuronExecution_free_fn)(NeuronExecution* execution);
  typedef int (*NeuronExecution_setInput_fn)(NeuronExecution* execution,
                                             int32_t index,
                                             const NeuronOperandType* type,
                                             const void* buffer,
                                             size_t length);
  typedef int (*NeuronExecution_setOutput_fn)(NeuronExecution* execution,
                                              int32_t index,
                                              const NeuronOperandType* type,
                                              void* buffer,
                                              size_t length);
  typedef int (*NeuronExecution_compute_fn)(NeuronExecution* execution);
  typedef int (*Neuron_getDeviceCount_fn)(uint32_t* numDevices);
  typedef int (*Neuron_getDevice_fn)(uint32_t devIndex, NeuronDevice** device);
  typedef int (*NeuronDevice_getName_fn)(const NeuronDevice* device,
                                         const char** name);

#define NEURON_ADAPTER_DECLARE_FUNCTION(name) name##_fn name;

  NEURON_ADAPTER_DECLARE_FUNCTION(Neuron_getVersion)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_create)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_free)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_finish)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_addOperand)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_setOperandValue)
  NEURON_ADAPTER_DECLARE_FUNCTION(
      NeuronModel_setOperandSymmPerChannelQuantParams)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_addOperation)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_addOperationExtension)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_identifyInputsAndOutputs)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_getSupportedOperations)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_relaxComputationFloat32toFloat16)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronModel_restoreFromCompiledNetwork)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_create)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_free)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_finish)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_setCaching)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_createForDevices)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_getCompiledNetworkSize)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronCompilation_storeCompiledNetwork)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronExecution_create)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronExecution_free)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronExecution_setInput)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronExecution_setOutput)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronExecution_compute)
  NEURON_ADAPTER_DECLARE_FUNCTION(Neuron_getDeviceCount)
  NEURON_ADAPTER_DECLARE_FUNCTION(Neuron_getDevice)
  NEURON_ADAPTER_DECLARE_FUNCTION(NeuronDevice_getName)
#undef NEURON_ADAPTER_DECLARE_FUNCTION

 private:
  NeuronAdapterWrapper();
  NeuronAdapterWrapper(const NeuronAdapterWrapper&) = delete;
  NeuronAdapterWrapper& operator=(const NeuronAdapterWrapper&) = delete;
  bool Initialize();
  bool initialized_{false};
  bool supported_{false};
  void* library_{nullptr};
};

inline int Neuron_getVersion_invoke(uint32_t* version) {
  return NeuronAdapterWrapper::Global().Neuron_getVersion(version);
}

inline int NeuronModel_create_invoke(NeuronModel** model) {
  return NeuronAdapterWrapper::Global().NeuronModel_create(model);
}

inline void NeuronModel_free_invoke(NeuronModel* model) {
  NeuronAdapterWrapper::Global().NeuronModel_free(model);
}

inline int NeuronModel_finish_invoke(NeuronModel* model) {
  return NeuronAdapterWrapper::Global().NeuronModel_finish(model);
}

inline int NeuronModel_addOperand_invoke(NeuronModel* model,
                                         const NeuronOperandType* type) {
  return NeuronAdapterWrapper::Global().NeuronModel_addOperand(model, type);
}

inline int NeuronModel_setOperandValue_invoke(NeuronModel* model,
                                              int32_t index,
                                              const void* buffer,
                                              size_t length) {
  return NeuronAdapterWrapper::Global().NeuronModel_setOperandValue(
      model, index, buffer, length);
}

inline int NeuronModel_setOperandSymmPerChannelQuantParams_invoke(
    NeuronModel* model,
    int32_t index,
    const NeuronSymmPerChannelQuantParams* channelQuant) {
  return NeuronAdapterWrapper::Global()
      .NeuronModel_setOperandSymmPerChannelQuantParams(
          model, index, channelQuant);
}

inline int NeuronModel_addOperation_invoke(NeuronModel* model,
                                           NeuronOperationType type,
                                           uint32_t inputCount,
                                           const uint32_t* inputs,
                                           uint32_t outputCount,
                                           const uint32_t* outputs) {
  return NeuronAdapterWrapper::Global().NeuronModel_addOperation(
      model, type, inputCount, inputs, outputCount, outputs);
}

inline int NeuronModel_addOperationExtension_invoke(NeuronModel* model,
                                                    const char* name,
                                                    const char* vendor,
                                                    const NeuronDevice* device,
                                                    uint32_t inputCount,
                                                    const uint32_t* inputs,
                                                    uint32_t outputCount,
                                                    const uint32_t* outputs) {
  return NeuronAdapterWrapper::Global().NeuronModel_addOperationExtension(
      model, name, vendor, device, inputCount, inputs, outputCount, outputs);
}

inline int NeuronModel_identifyInputsAndOutputs_invoke(
    NeuronModel* model,
    uint32_t inputCount,
    const uint32_t* inputs,
    uint32_t outputCount,
    const uint32_t* outputs) {
  return NeuronAdapterWrapper::Global().NeuronModel_identifyInputsAndOutputs(
      model, inputCount, inputs, outputCount, outputs);
}

inline int NeuronModel_getSupportedOperations_invoke(NeuronModel* model,
                                                     bool* supported,
                                                     uint32_t operationCount) {
  return NeuronAdapterWrapper::Global().NeuronModel_getSupportedOperations(
      model, supported, operationCount);
}

inline int NeuronModel_relaxComputationFloat32toFloat16_invoke(
    NeuronModel* model, bool allow) {
  return NeuronAdapterWrapper::Global()
      .NeuronModel_relaxComputationFloat32toFloat16(model, allow);
}

inline int NeuronModel_restoreFromCompiledNetwork_invoke(
    NeuronModel** model,
    NeuronCompilation** compilation,
    const void* buffer,
    const size_t size) {
  return NeuronAdapterWrapper::Global().NeuronModel_restoreFromCompiledNetwork(
      model, compilation, buffer, size);
}

inline int NeuronCompilation_create_invoke(NeuronModel* model,
                                           NeuronCompilation** compilation) {
  return NeuronAdapterWrapper::Global().NeuronCompilation_create(model,
                                                                 compilation);
}

inline void NeuronCompilation_free_invoke(NeuronCompilation* compilation) {
  NeuronAdapterWrapper::Global().NeuronCompilation_free(compilation);
}

inline int NeuronCompilation_finish_invoke(NeuronCompilation* compilation) {
  return NeuronAdapterWrapper::Global().NeuronCompilation_finish(compilation);
}

inline int NeuronCompilation_setCaching_invoke(NeuronCompilation* compilation,
                                               const char* cacheDir,
                                               const uint8_t* token) {
  return NeuronAdapterWrapper::Global().NeuronCompilation_setCaching(
      compilation, cacheDir, token);
}

inline int NeuronCompilation_createForDevices_invoke(
    NeuronModel* model,
    const NeuronDevice* const* devices,
    uint32_t numDevices,
    NeuronCompilation** compilation) {
  return NeuronAdapterWrapper::Global().NeuronCompilation_createForDevices(
      model, devices, numDevices, compilation);
}

inline int NeuronCompilation_getCompiledNetworkSize_invoke(
    NeuronCompilation* compilation, size_t* size) {
  return NeuronAdapterWrapper::Global()
      .NeuronCompilation_getCompiledNetworkSize(compilation, size);
}

inline int NeuronCompilation_storeCompiledNetwork_invoke(
    NeuronCompilation* compilation, void* buffer, const size_t size) {
  return NeuronAdapterWrapper::Global().NeuronCompilation_storeCompiledNetwork(
      compilation, buffer, size);
}

inline int NeuronExecution_create_invoke(NeuronCompilation* compilation,
                                         NeuronExecution** execution) {
  return NeuronAdapterWrapper::Global().NeuronExecution_create(compilation,
                                                               execution);
}

inline void NeuronExecution_free_invoke(NeuronExecution* execution) {
  NeuronAdapterWrapper::Global().NeuronExecution_free(execution);
}

inline int NeuronExecution_setInput_invoke(NeuronExecution* execution,
                                           int32_t index,
                                           const NeuronOperandType* type,
                                           const void* buffer,
                                           size_t length) {
  return NeuronAdapterWrapper::Global().NeuronExecution_setInput(
      execution, index, type, buffer, length);
}

inline int NeuronExecution_setOutput_invoke(NeuronExecution* execution,
                                            int32_t index,
                                            const NeuronOperandType* type,
                                            void* buffer,
                                            size_t length) {
  return NeuronAdapterWrapper::Global().NeuronExecution_setOutput(
      execution, index, type, buffer, length);
}

inline int NeuronExecution_compute_invoke(NeuronExecution* execution) {
  return NeuronAdapterWrapper::Global().NeuronExecution_compute(execution);
}

inline int Neuron_getDeviceCount_invoke(uint32_t* numDevices) {
  return NeuronAdapterWrapper::Global().Neuron_getDeviceCount(numDevices);
}

inline int Neuron_getDevice_invoke(uint32_t devIndex, NeuronDevice** device) {
  return NeuronAdapterWrapper::Global().Neuron_getDevice(devIndex, device);
}

inline int NeuronDevice_getName_invoke(const NeuronDevice* device,
                                       const char** name) {
  return NeuronAdapterWrapper::Global().NeuronDevice_getName(device, name);
}

}  // namespace mediatek_apu
}  // namespace nnadapter
