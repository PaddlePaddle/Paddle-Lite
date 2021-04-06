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

#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include <dlfcn.h>
#include <string>
#include <vector>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

NNAdapter& NNAdapter::Global() {
  static NNAdapter nnadapter;
  return nnadapter;
}

NNAdapter::NNAdapter() {
  CHECK(Init()) << "Failed to initialize NNAdapter library!";
}

bool NNAdapter::Init() {
  const std::vector<std::string> candidate_paths = {
      "libnnadapter.so",
  };
  std::string found_path = "Unknown";
  for (auto& candidate_path : candidate_paths) {
    library_ = dlopen(candidate_path.c_str(), RTLD_LAZY);
    if (library_ != nullptr) {
      found_path = candidate_path;
      break;
    }
    VLOG(4) << "Failed to load the NNAdapter library from " << candidate_path
            << "," << std::string(dlerror());
  }
  if (!library_) {
    return false;
  }
  VLOG(4) << "The NNAdapter library " << found_path << " is loaded.";

#define NNADAPTER_DLSYM(nnadapter_api)                                        \
  do {                                                                        \
    nnadapter_api##_ = (nnadapter_api##_Type)dlsym(library_, #nnadapter_api); \
    if (nnadapter_api##_ == nullptr) {                                        \
      LOG(WARNING) << "Cannot find the symbol " << #nnadapter_api             \
                   << " in libnnadapter.so! " << std::string(dlerror());      \
      return false;                                                           \
    }                                                                         \
    VLOG(4) << #nnadapter_api << " is loaded.";                               \
  } while (false)

  NNADAPTER_DLSYM(NNAdapterDevice_acquire);
  NNADAPTER_DLSYM(NNAdapterDevice_release);
  NNADAPTER_DLSYM(NNAdapterDevice_getName);
  NNADAPTER_DLSYM(NNAdapterDevice_getVendor);
  NNADAPTER_DLSYM(NNAdapterDevice_getType);
  NNADAPTER_DLSYM(NNAdapterDevice_getVersion);
  NNADAPTER_DLSYM(NNAdapterNetwork_create);
  NNADAPTER_DLSYM(NNAdapterNetwork_free);
  NNADAPTER_DLSYM(NNAdapterNetwork_addOperand);
  NNADAPTER_DLSYM(NNAdapterNetwork_setOperand);
  NNADAPTER_DLSYM(NNAdapterNetwork_addOperation);
  NNADAPTER_DLSYM(NNAdapterNetwork_setOperation);
  NNADAPTER_DLSYM(NNAdapterNetwork_identifyInputsAndOutputs);
  NNADAPTER_DLSYM(NNAapdterModel_createFromCache);
  NNADAPTER_DLSYM(NNAapdterModel_createFromNetwork);
  NNADAPTER_DLSYM(NNAapdterModel_free);
  NNADAPTER_DLSYM(NNAapdterModel_setCacheMode);
  NNADAPTER_DLSYM(NNAdapterModel_getCacheSize);
  NNADAPTER_DLSYM(NNAdapterModel_getCacheBuffer);
  NNADAPTER_DLSYM(NNAdapterExecution_create);
  NNADAPTER_DLSYM(NNAdapterExecution_free);
  NNADAPTER_DLSYM(NNAdapterExecution_setInput);
  NNADAPTER_DLSYM(NNAdapterExecution_setOutput);
  NNADAPTER_DLSYM(NNAdapterExecution_startCompute);
#undef NNADAPTER_DLSYM
  VLOG(4) << "Extract all of symbols from " << found_path << " done.";
  return true;
}

int32_t NNAdapter::NNAdapterDevice_acquire(const char* name,
                                           NNAdapterDevice** device) {
  CHECK(NNAdapterDevice_acquire_ != nullptr)
      << "NNAdapterDevice_acquire is nullptr!";
  return NNAdapterDevice_acquire_(name, device);
}

void NNAdapter::NNAdapterDevice_release(NNAdapterDevice* device) {
  CHECK(NNAdapterDevice_acquire_ != nullptr)
      << "NNAdapterDevice_acquire is nullptr!";
  return NNAdapterDevice_release_(device);
}

int32_t NNAdapter::NNAdapterDevice_getName(const NNAdapterDevice* device,
                                           const char** name) {
  CHECK(NNAdapterDevice_getName_ != nullptr)
      << "NNAdapterDevice_getName is nullptr!";
  return NNAdapterDevice_getName_(device, name);
}

int32_t NNAdapter::NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                                             const char** vendor) {
  CHECK(NNAdapterDevice_getVendor_ != nullptr)
      << "NNAdapterDevice_getVendor is nullptr!";
  return NNAdapterDevice_getVendor_(device, vendor);
}

int32_t NNAdapter::NNAdapterDevice_getType(const NNAdapterDevice* device,
                                           NNAdapterDeviceType* type) {
  CHECK(NNAdapterDevice_getType_ != nullptr)
      << "NNAdapterDevice_getType is nullptr!";
  return NNAdapterDevice_getType_(device, type);
}

int32_t NNAdapter::NNAdapterDevice_getVersion(const NNAdapterDevice* device,
                                              int32_t* version) {
  CHECK(NNAdapterDevice_getVersion_ != nullptr)
      << "NNAdapterDevice_getVersion is nullptr!";
  return NNAdapterDevice_getVersion_(device, version);
}

int32_t NNAdapter::NNAdapterNetwork_create(NNAdapterNetwork** network) {
  CHECK(NNAdapterNetwork_create_ != nullptr)
      << "NNAdapterNetwork_create is nullptr!";
  return NNAdapterNetwork_create_(network);
}

void NNAdapter::NNAdapterNetwork_free(NNAdapterNetwork* network) {
  CHECK(NNAdapterNetwork_free_ != nullptr)
      << "NNAdapterNetwork_free is nullptr!";
  NNAdapterNetwork_free_(network);
}

int32_t NNAdapter::NNAdapterNetwork_addOperand(NNAdapterNetwork* network,
                                               const NNAdapterOperandType* type,
                                               NNAdapterOperand** operand) {
  CHECK(NNAdapterNetwork_addOperand_ != nullptr)
      << "NNAdapterNetwork_addOperand is nullptr!";
  return NNAdapterNetwork_addOperand_(network, type, operand);
}

int32_t NNAdapter::NNAdapterNetwork_setOperand(NNAdapterOperand* operand,
                                               const void* buffer,
                                               size_t length) {
  CHECK(NNAdapterNetwork_setOperand_ != nullptr)
      << "NNAdapterNetwork_setOperand is nullptr!";
  return NNAdapterNetwork_setOperand_(operand, buffer, length);
}

int32_t NNAdapter::NNAdapterNetwork_addOperation(
    NNAdapterNetwork* network,
    NNAdapterOperationType type,
    NNAdapterOperation** operation) {
  CHECK(NNAdapterNetwork_addOperation_ != nullptr)
      << "NNAdapterNetwork_addOperation is nullptr!";
  return NNAdapterNetwork_addOperation_(network, type, operation);
}

int32_t NNAdapter::NNAdapterNetwork_setOperation(
    NNAdapterOperation* operation,
    uint32_t inputCount,
    const NNAdapterOperand* inputs,
    uint32_t outputCount,
    const NNAdapterOperand* outputs) {
  CHECK(NNAdapterNetwork_setOperation_ != nullptr)
      << "NNAdapterNetwork_setOperation is nullptr!";
  return NNAdapterNetwork_setOperation_(
      operation, inputCount, inputs, outputCount, outputs);
}

int32_t NNAdapter::NNAdapterNetwork_identifyInputsAndOutputs(
    NNAdapterNetwork* network,
    uint32_t inputCount,
    const NNAdapterOperand* inputs,
    uint32_t outputCount,
    const NNAdapterOperand* outputs) {
  CHECK(NNAdapterNetwork_identifyInputsAndOutputs_ != nullptr)
      << "NNAdapterNetwork_identifyInputsAndOutputs is nullptr!";
  return NNAdapterNetwork_identifyInputsAndOutputs_(
      network, inputCount, inputs, outputCount, outputs);
}

int32_t NNAdapter::NNAapdterModel_createFromCache(void* buffer,
                                                  const size_t size,
                                                  NNAdapterModel** model) {
  CHECK(NNAapdterModel_createFromCache_ != nullptr)
      << "NNAapdterModel_createFromCache is nullptr!";
  return NNAapdterModel_createFromCache_(buffer, size, model);
}

int32_t NNAdapter::NNAapdterModel_createFromNetwork(
    NNAdapterNetwork* network,
    const NNAdapterDevice* const* devices,
    uint32_t numDevices,
    NNAdapterModel** model) {
  CHECK(NNAapdterModel_createFromNetwork_ != nullptr)
      << "NNAapdterModel_createFromNetwork is nullptr!";
  return NNAapdterModel_createFromNetwork_(network, devices, numDevices, model);
}

void NNAdapter::NNAapdterModel_free(NNAdapterModel* model) {
  CHECK(NNAapdterModel_free_ != nullptr) << "NNAapdterModel_free is nullptr!";
  NNAapdterModel_free_(model);
}

int32_t NNAdapter::NNAapdterModel_setCacheMode(NNAdapterModel* model,
                                               const char* cacheDir,
                                               const uint8_t* token) {
  CHECK(NNAapdterModel_setCacheMode_ != nullptr)
      << "NNAapdterModel_setCacheMode is nullptr!";
  return NNAapdterModel_setCacheMode_(model, cacheDir, token);
}

int32_t NNAdapter::NNAdapterModel_getCacheSize(NNAdapterModel* model,
                                               size_t* size) {
  CHECK(NNAdapterModel_getCacheSize_ != nullptr)
      << "NNAdapterModel_getCacheSize is nullptr!";
  return NNAdapterModel_getCacheSize_(model, size);
}

int32_t NNAdapter::NNAdapterModel_getCacheBuffer(NNAdapterModel* model,
                                                 void* buffer,
                                                 const size_t size) {
  CHECK(NNAdapterModel_getCacheBuffer_ != nullptr)
      << "NNAdapterModel_getCacheBuffer is nullptr!";
  return NNAdapterModel_getCacheBuffer_(model, buffer, size);
}

int32_t NNAdapter::NNAdapterExecution_create(NNAdapterModel* model,
                                             NNAdapterExecution** execution) {
  CHECK(NNAdapterExecution_create_ != nullptr)
      << "NNAdapterExecution_create is nullptr!";
  return NNAdapterExecution_create_(model, execution);
}

void NNAdapter::NNAdapterExecution_free(NNAdapterExecution* execution) {
  CHECK(NNAdapterExecution_free_ != nullptr)
      << "NNAdapterExecution_free is nullptr!";
  return NNAdapterExecution_free_(execution);
}

int32_t NNAdapter::NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                               int32_t index,
                                               const NNAdapterOperandType* type,
                                               const void* buffer,
                                               size_t length) {
  CHECK(NNAdapterExecution_setInput_ != nullptr)
      << "NNAdapterExecution_setInput is nullptr!";
  return NNAdapterExecution_setInput_(execution, index, type, buffer, length);
}

int32_t NNAdapter::NNAdapterExecution_setOutput(
    NNAdapterExecution* execution,
    int32_t index,
    const NNAdapterOperandType* type,
    void* buffer,
    size_t length) {
  CHECK(NNAdapterExecution_setOutput_ != nullptr)
      << "NNAdapterExecution_setOutput is nullptr!";
  return NNAdapterExecution_setOutput_(execution, index, type, buffer, length);
}

int32_t NNAdapter::NNAdapterExecution_startCompute(
    NNAdapterExecution* execution) {
  CHECK(NNAdapterExecution_startCompute_ != nullptr)
      << "NNAdapterExecution_startCompute is nullptr!";
  return NNAdapterExecution_startCompute_(execution);
}

}  // namespace lite
}  // namespace paddle
