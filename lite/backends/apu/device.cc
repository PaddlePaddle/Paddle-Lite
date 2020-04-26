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

#include "lite/backends/apu/device.h"
#include <dlfcn.h>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace apu {

inline void* LoadFunc(void* libHandle, const char* name) {
  CHECK(libHandle != nullptr);
  CHECK(name != nullptr);
  void* fn = dlsym(libHandle, name);
  if (fn == nullptr) {
    LOG(WARNING) << "Unable to open Neuron Runtime function [" << name
                 << "] Because " << dlerror();
  }
  return fn;
}

NeuronCompilation* Device::Build(void* libHandle, NeuronModel* model) {
  typedef int (*NeuronCompilation_create)(NeuronModel * model,
                                          NeuronCompilation * *compilation);
  typedef void (*NeuronCompilation_free)(NeuronCompilation * compilation);
  typedef int (*NeuronCompilation_finish)(NeuronCompilation * compilation);

#define LOAD_FUNCTIONS(libHandle, FUNC_NAME, VARIABLE_NAME) \
  FUNC_NAME VARIABLE_NAME =                                 \
      reinterpret_cast<FUNC_NAME>(LoadFunc(libHandle, #FUNC_NAME));
  LOAD_FUNCTIONS(libHandle, NeuronCompilation_create, neuron_compilation_create)
  LOAD_FUNCTIONS(libHandle, NeuronCompilation_free, neuron_compilation_free)
  LOAD_FUNCTIONS(libHandle, NeuronCompilation_finish, neuron_compilation_finish)
#undef LOAD_FUNCTIONS

  int neuron_errCode = 0;
  NeuronCompilation* compilation = NULL;

  VLOG(3) << "[APU] Compile model";

  neuron_errCode = (*neuron_compilation_create)(model, &compilation);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] create compile failed! " << neuron_errCode;
    return nullptr;
  }

  neuron_errCode = (*neuron_compilation_finish)(compilation);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] compile failed! " << neuron_errCode;
    return nullptr;
  }

  VLOG(3) << "[APU] Build done";
  return compilation;
}

}  // namespace apu
}  // namespace lite
}  // namespace paddle
