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

NeuronCompilation* Device::Build(NeuronModel* model) {
  VLOG(3) << "[APU] Compile model";
  NeuronCompilation* compilation = NULL;
  int neuron_errCode = NeuronCompilation_create(model, &compilation);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] create compile failed! " << neuron_errCode;
    return nullptr;
  }
  neuron_errCode = NeuronCompilation_finish(compilation);
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
