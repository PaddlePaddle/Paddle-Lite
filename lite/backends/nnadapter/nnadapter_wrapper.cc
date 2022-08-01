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

}  // namespace lite
}  // namespace paddle
