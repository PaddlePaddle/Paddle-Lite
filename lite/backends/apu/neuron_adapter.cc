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

#include "lite/backends/apu/neuron_adapter.h"
#include <dlfcn.h>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
NeuronAdapter* NeuronAdapter::Global() {
  static NeuronAdapter adapter;
  return &adapter;
}

NeuronAdapter::NeuronAdapter() {
  CHECK(InitHandle()) << "Fail to initialize the Neuron Adapter library!";
  InitFunctions();
}

bool NeuronAdapter::InitHandle() {
  const std::vector<std::string> paths = {
    "libneuron_adapter.so",
#if defined(__aarch64__)
    "/vendor/lib64/libneuron_adapter.so",
    "/system/lib64/libneuron_adapter.so",
    "/system/vendor/lib64/libneuron_adapter.so",
#else
    "/vendor/lib/libneuron_adapter.so",
    "/system/lib/libneuron_adapter.so",
    "/system/vendor/lib/libneuron_adapter.so",
#endif
  };
  std::string target_lib = "Unknown";
  for (auto path : paths) {
    handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (handle_ != nullptr) {
      target_lib = path;
      break;
    }
  }
  VLOG(4) << "Load the Neuron Adapter library from " << target_lib;
  if (handle_ != nullptr) {
    return true;
  } else {
    return false;
  }
}

void NeuronAdapter::InitFunctions() {
  CHECK(handle_ != nullptr) << "The library handle can't be null!";

#define PADDLE_DLSYM(neuron_adapter_func)                                 \
  do {                                                                    \
    neuron_adapter_func##_ =                                              \
        (neuron_adapter_func##_Type)dlsym(handle_, #neuron_adapter_func); \
    if (neuron_adapter_func##_ == nullptr) {                              \
      LOG(FATAL) << "Cannot find the " << #neuron_adapter_func            \
                 << " symbol in libneuron_adapter.so!";                   \
      break;                                                              \
    }                                                                     \
    VLOG(4) << "Loaded the " << #neuron_adapter_func                      \
            << " symbol successfully.";                                   \
  } while (false)

  PADDLE_DLSYM(Neuron_getVersion);
  PADDLE_DLSYM(NeuronModel_create);
  PADDLE_DLSYM(NeuronModel_free);
  PADDLE_DLSYM(NeuronModel_finish);
  PADDLE_DLSYM(NeuronModel_addOperand);
  PADDLE_DLSYM(NeuronModel_setOperandValue);
  PADDLE_DLSYM(NeuronModel_setOperandSymmPerChannelQuantParams);
  PADDLE_DLSYM(NeuronModel_addOperation);
  PADDLE_DLSYM(NeuronModel_addOperationExtension);
  PADDLE_DLSYM(NeuronModel_identifyInputsAndOutputs);
  PADDLE_DLSYM(NeuronModel_restoreFromCompiledNetwork);
  PADDLE_DLSYM(NeuronCompilation_create);
  PADDLE_DLSYM(NeuronCompilation_free);
  PADDLE_DLSYM(NeuronCompilation_finish);
  PADDLE_DLSYM(NeuronCompilation_setCaching);
  PADDLE_DLSYM(NeuronCompilation_storeCompiledNetwork);
  PADDLE_DLSYM(NeuronCompilation_createForDevices);
  PADDLE_DLSYM(NeuronCompilation_getCompiledNetworkSize);
  PADDLE_DLSYM(NeuronExecution_create);
  PADDLE_DLSYM(NeuronExecution_free);
  PADDLE_DLSYM(NeuronExecution_setInput);
  PADDLE_DLSYM(NeuronExecution_setOutput);
  PADDLE_DLSYM(NeuronExecution_compute);
  PADDLE_DLSYM(Neuron_getDeviceCount);
  PADDLE_DLSYM(Neuron_getDevice);
  PADDLE_DLSYM(NeuronDevice_getName);
#undef PADDLE_DLSYM
}

}  // namespace lite
}  // namespace paddle

int Neuron_getVersion(uint32_t* version) {
  return paddle::lite::NeuronAdapter::Global()->Neuron_getVersion()(version);
}

int NeuronModel_create(NeuronModel** model) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_create()(model);
}

void NeuronModel_free(NeuronModel* model) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_free()(model);
}

int NeuronModel_finish(NeuronModel* model) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_finish()(model);
}

int NeuronModel_addOperand(NeuronModel* model, const NeuronOperandType* type) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_addOperand()(model,
                                                                         type);
}

int NeuronModel_setOperandValue(NeuronModel* model,
                                int32_t index,
                                const void* buffer,
                                size_t length) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_setOperandValue()(
      model, index, buffer, length);
}

int NeuronModel_setOperandSymmPerChannelQuantParams(
    NeuronModel* model,
    int32_t index,
    const NeuronSymmPerChannelQuantParams* channelQuant) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronModel_setOperandSymmPerChannelQuantParams()(
          model, index, channelQuant);
}

int NeuronModel_addOperation(NeuronModel* model,
                             NeuronOperationType type,
                             uint32_t inputCount,
                             const uint32_t* inputs,
                             uint32_t outputCount,
                             const uint32_t* outputs) {
  return paddle::lite::NeuronAdapter::Global()->NeuronModel_addOperation()(
      model, type, inputCount, inputs, outputCount, outputs);
}

int NeuronModel_addOperationExtension(NeuronModel* model,
                                      const char* name,
                                      const char* vendor,
                                      const NeuronDevice* device,
                                      uint32_t inputCount,
                                      const uint32_t* inputs,
                                      uint32_t outputCount,
                                      const uint32_t* outputs) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronModel_addOperationExtension()(model,
                                            name,
                                            vendor,
                                            device,
                                            inputCount,
                                            inputs,
                                            outputCount,
                                            outputs);
}

int NeuronModel_identifyInputsAndOutputs(NeuronModel* model,
                                         uint32_t inputCount,
                                         const uint32_t* inputs,
                                         uint32_t outputCount,
                                         const uint32_t* outputs) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronModel_identifyInputsAndOutputs()(
          model, inputCount, inputs, outputCount, outputs);
}

int NeuronModel_restoreFromCompiledNetwork(NeuronModel** model,
                                           NeuronCompilation** compilation,
                                           const void* buffer,
                                           const size_t size) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronModel_restoreFromCompiledNetwork()(
          model, compilation, buffer, size);
}

int NeuronCompilation_create(NeuronModel* model,
                             NeuronCompilation** compilation) {
  return paddle::lite::NeuronAdapter::Global()->NeuronCompilation_create()(
      model, compilation);
}

void NeuronCompilation_free(NeuronCompilation* compilation) {
  return paddle::lite::NeuronAdapter::Global()->NeuronCompilation_free()(
      compilation);
}

int NeuronCompilation_finish(NeuronCompilation* compilation) {
  return paddle::lite::NeuronAdapter::Global()->NeuronCompilation_finish()(
      compilation);
}

int NeuronCompilation_setCaching(NeuronCompilation* compilation,
                                 const char* cacheDir,
                                 const uint8_t* token) {
  return paddle::lite::NeuronAdapter::Global()->NeuronCompilation_setCaching()(
      compilation, cacheDir, token);
}

int NeuronCompilation_storeCompiledNetwork(NeuronCompilation* compilation,
                                           void* buffer,
                                           const size_t size) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronCompilation_storeCompiledNetwork()(compilation, buffer, size);
}

int NeuronCompilation_getCompiledNetworkSize(NeuronCompilation* compilation,
                                             size_t* size) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronCompilation_getCompiledNetworkSize()(compilation, size);
}

int NeuronCompilation_createForDevices(NeuronModel* model,
                                       const NeuronDevice* const* devices,
                                       uint32_t numDevices,
                                       NeuronCompilation** compilation) {
  return paddle::lite::NeuronAdapter::Global()
      ->NeuronCompilation_createForDevices()(
          model, devices, numDevices, compilation);
}

int NeuronExecution_create(NeuronCompilation* compilation,
                           NeuronExecution** execution) {
  return paddle::lite::NeuronAdapter::Global()->NeuronExecution_create()(
      compilation, execution);
}

void NeuronExecution_free(NeuronExecution* execution) {
  return paddle::lite::NeuronAdapter::Global()->NeuronExecution_free()(
      execution);
}

int NeuronExecution_setInput(NeuronExecution* execution,
                             int32_t index,
                             const NeuronOperandType* type,
                             const void* buffer,
                             size_t length) {
  return paddle::lite::NeuronAdapter::Global()->NeuronExecution_setInput()(
      execution, index, type, buffer, length);
}

int NeuronExecution_setOutput(NeuronExecution* execution,
                              int32_t index,
                              const NeuronOperandType* type,
                              void* buffer,
                              size_t length) {
  return paddle::lite::NeuronAdapter::Global()->NeuronExecution_setOutput()(
      execution, index, type, buffer, length);
}

int NeuronExecution_compute(NeuronExecution* execution) {
  return paddle::lite::NeuronAdapter::Global()->NeuronExecution_compute()(
      execution);
}

int Neuron_getDeviceCount(uint32_t* numDevices) {
  return paddle::lite::NeuronAdapter::Global()->Neuron_getDeviceCount()(
      numDevices);
}

int Neuron_getDevice(uint32_t devIndex, NeuronDevice** device) {
  return paddle::lite::NeuronAdapter::Global()->Neuron_getDevice()(devIndex,
                                                                   device);
}

int NeuronDevice_getName(const NeuronDevice* device, const char** name) {
  return paddle::lite::NeuronAdapter::Global()->NeuronDevice_getName()(device,
                                                                       name);
}
