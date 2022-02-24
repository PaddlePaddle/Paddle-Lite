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

#include "driver/mediatek_apu/neuron_adapter_wrapper.h"
#include <dlfcn.h>
#include <string>
#include <vector>
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

NeuronAdapterWrapper& NeuronAdapterWrapper::Global() {
  static NeuronAdapterWrapper wrapper;
  return wrapper;
}

NeuronAdapterWrapper::NeuronAdapterWrapper() {
  if (!initialized_) {
    supported_ = Initialize();
    initialized_ = true;
  }
}

bool NeuronAdapterWrapper::Initialize() {
  const std::vector<std::string> candidate_paths = {
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
  std::string found_path = "Unknown";
  for (auto& candidate_path : candidate_paths) {
    library_ = dlopen(candidate_path.c_str(), RTLD_LAZY);
    if (library_ != nullptr) {
      found_path = candidate_path;
      break;
    }
    NNADAPTER_VLOG(4) << "Failed to load the Neuron Adapter library from "
                      << candidate_path << "," << std::string(dlerror());
  }
  if (!library_) {
    return false;
  }
  NNADAPTER_VLOG(4) << "The Neuron Adapter library " << found_path
                    << " is loaded.";

#define NEURON_ADAPTER_LOAD_FUNCTION(name)                                    \
  do {                                                                        \
    name = (name##_fn)dlsym(library_, #name);                                 \
    if (name == nullptr) {                                                    \
      NNADAPTER_LOG(WARNING) << "Cannot find the symbol " << #name << " in "  \
                             << found_path << ". " << std::string(dlerror()); \
      return false;                                                           \
    }                                                                         \
    NNADAPTER_VLOG(4) << #name << " is loaded.";                              \
  } while (false);

  NEURON_ADAPTER_LOAD_FUNCTION(Neuron_getVersion)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_create)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_free)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_finish)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_addOperand)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_setOperandValue)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_setOperandSymmPerChannelQuantParams)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_addOperation)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_addOperationExtension)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_identifyInputsAndOutputs)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_getSupportedOperations)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_relaxComputationFloat32toFloat16)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronModel_restoreFromCompiledNetwork)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_create)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_free)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_finish)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_setCaching)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_createForDevices)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_getCompiledNetworkSize)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronCompilation_storeCompiledNetwork)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronExecution_create)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronExecution_free)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronExecution_setInput)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronExecution_setOutput)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronExecution_compute)
  NEURON_ADAPTER_LOAD_FUNCTION(Neuron_getDeviceCount)
  NEURON_ADAPTER_LOAD_FUNCTION(Neuron_getDevice)
  NEURON_ADAPTER_LOAD_FUNCTION(NeuronDevice_getName)
#undef NEURON_ADAPTER_LOAD_FUNCTION
  NNADAPTER_VLOG(4) << "Extract all of symbols from " << found_path << " done.";
  return true;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
