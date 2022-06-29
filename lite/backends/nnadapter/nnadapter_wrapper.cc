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
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

NNAdapterWrapper& NNAdapterWrapper::Global() {
  static NNAdapterWrapper wrapper;
  return wrapper;
}

NNAdapterWrapper::NNAdapterWrapper() {
  if (!initialized_) {
    supported_ = Initialize();
    initialized_ = true;
  }
}

NNAdapterWrapper::~NNAdapterWrapper() {
  if (library_) {
    dlclose(library_);
    library_ = nullptr;
  }
}

bool NNAdapterWrapper::Initialize() {
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

#define NNADAPTER_LOAD_FUNCTION(name)                               \
  do {                                                              \
    name = (name##_fn)dlsym(library_, #name);                       \
    if (name == nullptr) {                                          \
      LOG(WARNING) << "Cannot find the symbol " << #name << " in "  \
                   << found_path << ". " << std::string(dlerror()); \
      return false;                                                 \
    }                                                               \
    VLOG(4) << #name << " is loaded.";                              \
  } while (false);

  NNADAPTER_LOAD_FUNCTION(NNAdapter_getVersion)
  NNADAPTER_LOAD_FUNCTION(NNAdapter_getDeviceCount)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_acquire)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_release)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_getName)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_getVendor)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_getType)
  NNADAPTER_LOAD_FUNCTION(NNAdapterDevice_getVersion)
  NNADAPTER_LOAD_FUNCTION(NNAdapterContext_create)
  NNADAPTER_LOAD_FUNCTION(NNAdapterContext_destroy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterMemory_create)
  NNADAPTER_LOAD_FUNCTION(NNAdapterMemory_destroy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterMemory_copy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_create)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_destroy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_finish)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_addOperand)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_setOperandValue)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_getOperandType)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_addOperation)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_identifyInputsAndOutputs)
  NNADAPTER_LOAD_FUNCTION(NNAdapterModel_getSupportedOperations)
  NNADAPTER_LOAD_FUNCTION(NNAdapterCompilation_create)
  NNADAPTER_LOAD_FUNCTION(NNAdapterCompilation_destroy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterCompilation_finish)
  NNADAPTER_LOAD_FUNCTION(NNAdapterCompilation_queryInputsAndOutputs)
  NNADAPTER_LOAD_FUNCTION(NNAdapterExecution_create)
  NNADAPTER_LOAD_FUNCTION(NNAdapterExecution_destroy)
  NNADAPTER_LOAD_FUNCTION(NNAdapterExecution_setInput)
  NNADAPTER_LOAD_FUNCTION(NNAdapterExecution_setOutput)
  NNADAPTER_LOAD_FUNCTION(NNAdapterExecution_compute)
#undef NNADAPTER_LOAD_FUNCTION
  VLOG(4) << "Extract all of symbols from " << found_path << " done.";
  return true;
}

NNAdapterRuntimeInstance::~NNAdapterRuntimeInstance() {
  DestroyContextAndReleaseDevices();
}

bool NNAdapterRuntimeInstance::AcquireDevicesAndCreateContext(
    const std::vector<std::string>& device_names,
    const std::string& context_properties,
    int (*context_callback)(int event_id, void* user_data)) {
  CHECK(!IsValid()) << "Device context can only be created once!";
  CHECK_GT(device_names.size(), 0) << "No device specified.";
  devices_.clear();
  device_names_.clear();
  device_vendors_.clear();
  device_types_.clear();
  for (const auto& device_name : device_names) {
    NNAdapterDevice* device = nullptr;
    int result = NNAdapterDevice_acquire_invoke(device_name.c_str(), &device);
    bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
    if (found) {
      const char* name = nullptr;
      NNAdapterDevice_getName_invoke(device, &name);
      const char* vendor = nullptr;
      NNAdapterDevice_getVendor_invoke(device, &vendor);
      NNAdapterDeviceType type = 0;
      NNAdapterDevice_getType_invoke(device, &type);
      int32_t version = 0;
      NNAdapterDevice_getVersion_invoke(device, &version);
      VLOG(3) << "NNAdapter device " << name << ": vendor=" << vendor
              << " type=" << type << " version=" << version;
      devices_.push_back(device);
      device_names_.push_back(name);
      device_vendors_.push_back(vendor);
      device_types_.push_back(type);
    }
  }
  CHECK_GT(devices_.size(), 0) << "No device found.";
  VLOG(3) << "NNAdapter context_properties: " << context_properties;
  VLOG(3) << "NNAdapter context_callback: " << context_callback;
  // Create a context shared by multiple devices
  return NNAdapterContext_create_invoke(devices_.data(),
                                        devices_.size(),
                                        context_properties.c_str(),
                                        context_callback,
                                        &context_) == NNADAPTER_NO_ERROR;
}

void NNAdapterRuntimeInstance::DestroyContextAndReleaseDevices() {
  if (context_) {
    NNAdapterContext_destroy_invoke(context_);
  }
  for (auto device : devices_) {
    if (device) {
      NNAdapterDevice_release_invoke(device);
    }
  }
}

bool CheckNNAdapterDeviceAvailable(const std::string& device_name) {
  NNAdapterDevice* device = nullptr;
  int result = NNAdapterDevice_acquire_invoke(device_name.c_str(), &device);
  bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
  if (found) {
    NNAdapterDevice_release_invoke(device);
  }
  return found;
}

}  // namespace lite
}  // namespace paddle
