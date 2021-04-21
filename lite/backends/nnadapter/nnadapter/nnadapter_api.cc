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

#include "nnadapter_api.h"  // NOLINT
#include <stdlib.h>
#include <vector>
#include "nnadapter_logging.h"  // NOLINT
#include "nnadapter_runtime.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

NNADAPTER_EXPORT int NNAdapterDevice_acquire(const char* name,
                                             NNAdapterDevice** device) {
  if (!name || !device) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = new nnadapter::runtime::Device(name);
  if (d == nullptr) {
    *device = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  if (!d->hasDriver()) {
    delete d;
    NNADAPTER_LOG(ERROR) << "The NNAdapter driver of '" << name
                         << "' is not initialized.";
    return NNADAPTER_DEVICE_NOT_FOUND;
  }
  *device = reinterpret_cast<NNAdapterDevice*>(d);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterDevice_release(NNAdapterDevice* device) {
  if (!device) {
    auto d = reinterpret_cast<nnadapter::runtime::Device*>(device);
    delete d;
  }
}

NNADAPTER_EXPORT int NNAdapterDevice_getName(const NNAdapterDevice* device,
                                             const char** name) {
  if (!device || !name) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *name = d->getName();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                                               const char** vendor) {
  if (!device || !vendor) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *vendor = d->getVendor();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getType(const NNAdapterDevice* device,
                                             NNAdapterDeviceType* type) {
  if (!device || !type) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *type = d->getType();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getVersion(const NNAdapterDevice* device,
                                                int32_t* version) {
  if (!device || !version) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *version = d->getVersion();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterGraph_create(NNAdapterGraph** graph) {
  if (!graph) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = new nnadapter::runtime::Graph();
  if (g == nullptr) {
    *graph = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *graph = reinterpret_cast<NNAdapterGraph*>(g);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterGraph_destroy(NNAdapterGraph* graph) {
  if (!graph) {
    auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
    delete g;
  }
}

NNADAPTER_EXPORT int NNAdapterGraph_finish(NNAdapterGraph* graph) {
  if (!graph) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
  return g->finish();
}

NNADAPTER_EXPORT int NNAdapterGraph_addOperand(NNAdapterGraph* graph,
                                               const NNAdapterOperandType* type,
                                               NNAdapterOperand** operand) {
  if (!graph || !type || !operand) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
  nnadapter::driver::Operand* o = nullptr;
  int result = g->addOperand(*type, &o);
  if (result == NNADAPTER_NO_ERROR) {
    *operand = reinterpret_cast<NNAdapterOperand*>(o);
  }
  return result;
}

NNADAPTER_EXPORT int NNAdapterGraph_setOperand(NNAdapterOperand* operand,
                                               void* buffer,
                                               size_t length) {
  if (!operand || !buffer || !length) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto o = reinterpret_cast<nnadapter::driver::Operand*>(operand);
  o->buffer = malloc(length);
  if (!o->buffer) return NNADAPTER_OUT_OF_MEMORY;
  memcpy(o->buffer, buffer, length);
  o->length = length;
  o->type.lifetime = NNADAPTER_CONSTANT;
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterGraph_addOperation(
    NNAdapterGraph* graph,
    NNAdapterOperationType type,
    NNAdapterOperation** operation) {
  if (!graph || !operation) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
  nnadapter::driver::Operation* o = nullptr;
  int result = g->addOperation(type, &o);
  if (result == NNADAPTER_NO_ERROR) {
    *operation = reinterpret_cast<NNAdapterOperation*>(o);
  }
  return result;
}

NNADAPTER_EXPORT int NNAdapterGraph_setOperation(NNAdapterOperation** operation,
                                                 uint32_t inputCount,
                                                 NNAdapterOperand** inputs,
                                                 uint32_t outputCount,
                                                 NNAdapterOperand** outputs) {
  if (!operation) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto o = reinterpret_cast<nnadapter::driver::Operation*>(operation);
  o->inputs.resize(inputCount);
  for (uint32_t i = 0; i < inputCount; i++) {
    o->inputs[i] = reinterpret_cast<nnadapter::driver::Operand*>(inputs[i]);
  }
  o->outputs.resize(outputCount);
  for (uint32_t i = 0; i < outputCount; i++) {
    o->outputs[i] = reinterpret_cast<nnadapter::driver::Operand*>(outputs[i]);
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterGraph_identifyInputsAndOutputs(
    NNAdapterGraph* graph,
    uint32_t inputCount,
    NNAdapterOperand** inputs,
    uint32_t outputCount,
    NNAdapterOperand** outputs) {
  if (!graph || !outputs || !outputCount) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
  auto is = reinterpret_cast<nnadapter::driver::Operand**>(inputs);
  auto os = reinterpret_cast<nnadapter::driver::Operand**>(outputs);
  return g->identifyInputsAndOutputs(inputCount, is, outputCount, os);
}

NNADAPTER_EXPORT int NNAdapterModel_createFromGraph(NNAdapterGraph* graph,
                                                    NNAdapterDevice** devices,
                                                    uint32_t numDevices,
                                                    NNAdapterModel** model) {
  if (!graph || !devices || !model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto g = reinterpret_cast<nnadapter::runtime::Graph*>(graph);
  std::vector<nnadapter::runtime::Device*> ds;
  for (uint32_t i = 0; i < numDevices; i++) {
    ds.push_back(reinterpret_cast<nnadapter::runtime::Device*>(devices[i]));
  }
  auto m = new nnadapter::runtime::Model(g, ds);
  if (m == nullptr) {
    *model = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *model = reinterpret_cast<NNAdapterModel*>(m);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterModel_createFromCache(
    void* buffer,
    size_t length,
    uint32_t inputCount,
    const NNAdapterOperandType** inputTypes,
    uint32_t outputCount,
    const NNAdapterOperandType** outputTypes,
    NNAdapterDevice** devices,
    uint32_t numDevices,
    NNAdapterModel** model) {
  if (!buffer || !length || !model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  std::vector<nnadapter::runtime::Device*> ds;
  for (uint32_t i = 0; i < numDevices; i++) {
    ds.push_back(reinterpret_cast<nnadapter::runtime::Device*>(devices[i]));
  }
  auto m = new nnadapter::runtime::Model(
      buffer, length, inputCount, inputTypes, outputCount, outputTypes, ds);
  if (m == nullptr) {
    *model = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *model = reinterpret_cast<NNAdapterModel*>(m);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterModel_destroy(NNAdapterModel* model) {
  if (!model) {
    auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
    delete m;
  }
}

NNADAPTER_EXPORT int NNAdapterModel_finish(NNAdapterModel* model) {
  if (!model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  return m->finish();
}

NNADAPTER_EXPORT int NNAdapterModel_setCaching(NNAdapterModel* model,
                                               const char* cacheDir,
                                               const uint8_t* token) {
  if (!model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  return m->setCaching(cacheDir, token);
}

NNADAPTER_EXPORT int NNAdapterExecution_create(NNAdapterModel* model,
                                               NNAdapterExecution** execution) {
  if (!model || !execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  auto e = new nnadapter::runtime::Execution(m);
  if (e == nullptr) {
    *execution = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *execution = reinterpret_cast<NNAdapterExecution*>(e);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterExecution_destroy(
    NNAdapterExecution* execution) {
  if (!execution) {
    auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
    delete e;
  }
}

NNADAPTER_EXPORT int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                                 int32_t index,
                                                 const uint32_t* dimensions,
                                                 uint32_t dimensionCount,
                                                 void* buffer,
                                                 size_t length) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  return e->setInput(index, dimensions, dimensionCount, buffer, length);
}

NNADAPTER_EXPORT int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                                  int32_t index,
                                                  const uint32_t* dimensions,
                                                  uint32_t dimensionCount,
                                                  void* buffer,
                                                  size_t length) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  return e->setOutput(index, dimensions, dimensionCount, buffer, length);
}

NNADAPTER_EXPORT int NNAdapterExecution_run(NNAdapterExecution* execution,
                                            NNAdapterEvent** event) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  int result;
  if (event) {
    nnadapter::runtime::Event* n = nullptr;
    result = e->run(&n);
    *event = reinterpret_cast<NNAdapterEvent*>(n);
  } else {
    result = e->run(nullptr);
  }
  return result;
}

NNADAPTER_EXPORT int NNAdapterEvent_wait(NNAdapterEvent* event) {
  if (!event) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto n = reinterpret_cast<nnadapter::runtime::Event*>(event);
  n->wait();
  return n->getStatus();
}

NNADAPTER_EXPORT void NNAdapterEvent_destroy(NNAdapterEvent* event) {
  if (event) {
    auto n = reinterpret_cast<nnadapter::runtime::Event*>(event);
    n->wait();
    delete n;
  }
}

#ifdef __cplusplus
}
#endif
