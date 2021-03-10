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

#include "nnadapter_runtime.h"  // NOLINT

int NNAdapterAcquireDevice(uint32_t* devIndexOrNum, NNAdapterDevice** device) {
  return NNADAPTER_NO_ERROR;
}

void NNAdapterReleaseDevice(NNAdapterDevice* device) {}

int NNAdapterDeviceGetName(const NNAdapterDevice* device, const char** name) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterDeviceGetType(const NNAdapterDevice* device, int32_t* type) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterDeviceGetAPIVersion(const NNAdapterDevice* device,
                                 const char** version) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterDeviceGetDriverVersion(const NNAdapterDevice* device,
                                    const char** version) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterCreateNetwork(NNAdapterNetwork** network) {
  if (!network) {
    return NNADAPTER_INVALID_OBJECT;
  }
  NNAdapterNetworkBuilder* n = new NNAdapterNetworkBuilder();
  if (n == nullptr) {
    *network = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *network = reinterpret_cast<NNAdapterNetwork*>(n);
  return NNADAPTER_NO_ERROR;
}

void NNAdapterDestroyNetwork(NNAdapterNetwork* network) {
  if (!network) {
    NNAdapterNetworkBuilder* n =
        reinterpret_cast<NNAdapterNetworkBuilder*>(network);
    delete n;
  }
}

int NNAdapterNetworkAddOperand(NNAdapterNetwork* network,
                               const NNAdapterOperandType* type,
                               NNAdapterOperand** operand) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterNetworkSetOperand(NNAdapterOperand* operand,
                               const void* buffer,
                               size_t length) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterNetworkAddOperation(NNAdapterNetwork* network,
                                 NNAdapterOperationType type,
                                 NNAdapterOperation** operation) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterNetworkSetOperation(NNAdapterOperation* operation,
                                 uint32_t inputCount,
                                 const NNAdapterOperand* inputs,
                                 uint32_t outputCount,
                                 const NNAdapterOperand* outputs) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterNetworkIdentifyInputsAndOutputs(NNAdapterNetwork* network,
                                             uint32_t inputCount,
                                             const NNAdapterOperand* inputs,
                                             uint32_t outputCount,
                                             const NNAdapterOperand* outputs) {
  return NNADAPTER_NO_ERROR;
}

int NNAapdterCreateModelFromCache(void* buffer,
                                  const size_t size,
                                  NNAdapterModel** model) {
  return NNADAPTER_NO_ERROR;
}

int NNAapdterCreateModelFromNetwork(NNAdapterNetwork* network,
                                    const NNAdapterDevice* const* devices,
                                    uint32_t numDevices,
                                    NNAdapterModel** model) {
  return NNADAPTER_NO_ERROR;
}

void NNAapdterDestroyModel(NNAdapterModel* model) {}

int NNAapdterModelSetCacheMode(NNAdapterModel* model,
                               const char* cacheDir,
                               const uint8_t* token) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterModelGetCacheSize(NNAdapterModel* model, size_t* size) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterModelGetCacheBuffer(NNAdapterModel* model,
                                 void* buffer,
                                 const size_t size) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterCreateExecution(NNAdapterModel* model,
                             NNAdapterExecution** execution) {
  return NNADAPTER_NO_ERROR;
}

void NNAdapterDestroyExecution(NNAdapterExecution* execution) {}

int NNAdapterExecutionSetInput(NNAdapterExecution* execution,
                               int32_t index,
                               const NNAdapterOperandType* type,
                               const void* buffer,
                               size_t length) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterExecutionSetOutput(NNAdapterExecution* execution,
                                int32_t index,
                                const NNAdapterOperandType* type,
                                void* buffer,
                                size_t length) {
  return NNADAPTER_NO_ERROR;
}

int NNAdapterExecutionStartCompute(NNAdapterExecution* execution) {
  return NNADAPTER_NO_ERROR;
}
