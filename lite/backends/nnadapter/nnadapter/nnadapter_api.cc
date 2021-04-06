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

#include "nnadapter_logging.h"  // NOLINT
#include "nnadapter_runtime.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

NNADAPTER_EXPORT int32_t NNAdapterDevice_acquire(const char* name,
                                                 NNAdapterDevice** device) {
  if (!name || !device) {
    return NNADAPTER_INVALID_OBJECT;
  }
  nnadapter::runtime::Device* d = new nnadapter::runtime::Device(name);
  if (d == nullptr) {
    *device = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  if (!d->hasDriver()) {
    delete d;
    NNADAPTER_LOG(ERROR) << "The NNAdapter driver of '" << name
                         << "' is not initialized.";
    return NNADAPTER_NOT_INITIALIZED;
  }
  *device = reinterpret_cast<NNAdapterDevice*>(d);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterDevice_release(NNAdapterDevice* device) {
  if (!device) {
    nnadapter::runtime::Device* d =
        reinterpret_cast<nnadapter::runtime::Device*>(device);
    delete d;
  }
}

NNADAPTER_EXPORT int32_t NNAdapterDevice_getName(const NNAdapterDevice* device,
                                                 const char** name) {
  if (!device || !name) {
    return NNADAPTER_INVALID_OBJECT;
  }
  const nnadapter::runtime::Device* d =
      reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *name = d->getName();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterDevice_getVendor(const NNAdapterDevice* device, const char** vendor) {
  if (!device || !vendor) {
    return NNADAPTER_INVALID_OBJECT;
  }
  const nnadapter::runtime::Device* d =
      reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *vendor = d->getVendor();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterDevice_getType(const NNAdapterDevice* device,
                                                 NNAdapterDeviceType* type) {
  if (!device || !type) {
    return NNADAPTER_INVALID_OBJECT;
  }
  const nnadapter::runtime::Device* d =
      reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *type = d->getType();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterDevice_getVersion(const NNAdapterDevice* device, int32_t* version) {
  if (!device || !version) {
    return NNADAPTER_INVALID_OBJECT;
  }
  const nnadapter::runtime::Device* d =
      reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *version = d->getVersion();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterNetwork_create(NNAdapterNetwork** network) {
  if (!network) {
    return NNADAPTER_INVALID_OBJECT;
  }
  nnadapter::runtime::Network* n = new nnadapter::runtime::Network();
  if (n == nullptr) {
    *network = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *network = reinterpret_cast<NNAdapterNetwork*>(n);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterNetwork_free(NNAdapterNetwork* network) {
  if (!network) {
    nnadapter::runtime::Network* n =
        reinterpret_cast<nnadapter::runtime::Network*>(network);
    delete n;
  }
}

NNADAPTER_EXPORT int32_t
NNAdapterNetwork_addOperand(NNAdapterNetwork* network,
                            const NNAdapterOperandType* type,
                            NNAdapterOperand** operand) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterNetwork_setOperand(NNAdapterOperand* operand,
                                                     const void* buffer,
                                                     size_t length) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterNetwork_addOperation(NNAdapterNetwork* network,
                              NNAdapterOperationType type,
                              NNAdapterOperation** operation) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterNetwork_setOperation(NNAdapterOperation* operation,
                              uint32_t inputCount,
                              const NNAdapterOperand* inputs,
                              uint32_t outputCount,
                              const NNAdapterOperand* outputs) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterNetwork_identifyInputsAndOutputs(NNAdapterNetwork* network,
                                          uint32_t inputCount,
                                          const NNAdapterOperand* inputs,
                                          uint32_t outputCount,
                                          const NNAdapterOperand* outputs) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAapdterModel_createFromCache(
    void* buffer, const size_t size, NNAdapterModel** model) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAapdterModel_createFromNetwork(NNAdapterNetwork* network,
                                 const NNAdapterDevice* const* devices,
                                 uint32_t numDevices,
                                 NNAdapterModel** model) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAapdterModel_free(NNAdapterModel* model) {}

NNADAPTER_EXPORT int32_t NNAapdterModel_setCacheMode(NNAdapterModel* model,
                                                     const char* cacheDir,
                                                     const uint8_t* token) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterModel_getCacheSize(NNAdapterModel* model,
                                                     size_t* size) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterModel_getCacheBuffer(NNAdapterModel* model,
                                                       void* buffer,
                                                       const size_t size) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t NNAdapterExecution_create(
    NNAdapterModel* model, NNAdapterExecution** execution) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterExecution_free(NNAdapterExecution* execution) {}

NNADAPTER_EXPORT int32_t
NNAdapterExecution_setInput(NNAdapterExecution* execution,
                            int32_t index,
                            const NNAdapterOperandType* type,
                            const void* buffer,
                            size_t length) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                             int32_t index,
                             const NNAdapterOperandType* type,
                             void* buffer,
                             size_t length) {
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int32_t
NNAdapterExecution_startCompute(NNAdapterExecution* execution) {
  return NNADAPTER_NO_ERROR;
}

#ifdef __cplusplus
}
#endif
