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

#include "nnadapter.h"  // NOLINT
#include <string>
#include <vector>
#include "runtime/compilation.h"
#include "runtime/context.h"
#include "runtime/device.h"
#include "runtime/execution.h"
#include "runtime/model.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"

#define REGISTER_OPERATION(__op_type__,            \
                           __validate_func_name__, \
                           __prepare_func_name__,  \
                           __execute_func_name__,  \
                           ...)                    \
  extern int __prepare_func_name__(nnadapter::core::Operation* operation);
namespace nnadapter {
namespace operation {
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_OPERATION_ALL_H__
}  // namespace operation
}  // namespace nnadapter
#undef REGISTER_OPERATION

#ifdef __cplusplus
extern "C" {
#endif

NNADAPTER_EXPORT int NNAdapter_getVersion(uint32_t* version) {
  if (!version) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *version = NNADAPTER_VERSION;
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapter_getDeviceCount(uint32_t* numDevices) {
  if (!numDevices) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  // TODO(hong19860320) Find all driver libraries and register them to
  // DeviceManager.
  NNADAPTER_LOG(ERROR) << "Unsupported.";
  return NNADAPTER_NO_ERROR;
}

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
  if (!d->IsValid()) {
    delete d;
    NNADAPTER_LOG(ERROR) << "The NNAdapter driver of '" << name
                         << "' is not initialized.";
    return NNADAPTER_DEVICE_NOT_FOUND;
  }
  *device = reinterpret_cast<NNAdapterDevice*>(d);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterDevice_release(NNAdapterDevice* device) {
  if (device) {
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
  *name = d->GetName();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                                               const char** vendor) {
  if (!device || !vendor) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *vendor = d->GetVendor();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getType(const NNAdapterDevice* device,
                                             NNAdapterDeviceType* type) {
  if (!device || !type) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *type = d->GetType();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterDevice_getVersion(const NNAdapterDevice* device,
                                                int32_t* version) {
  if (!device || !version) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto d = reinterpret_cast<const nnadapter::runtime::Device*>(device);
  *version = d->GetVersion();
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterContext_create(NNAdapterDevice** devices,
                                             uint32_t num_devices,
                                             const char* properties,
                                             int (*callback)(int event_id,
                                                             void* user_data),
                                             NNAdapterContext** context) {
  if (!devices || !num_devices || !context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  std::vector<nnadapter::runtime::Device*> ds;
  for (uint32_t i = 0; i < num_devices; i++) {
    ds.push_back(reinterpret_cast<nnadapter::runtime::Device*>(devices[i]));
  }
  auto x =
      new nnadapter::runtime::Context(ds, std::string(properties), callback);
  if (x == nullptr) {
    *context = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<NNAdapterContext*>(x);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterContext_destroy(NNAdapterContext* context) {
  if (context) {
    auto x = reinterpret_cast<nnadapter::runtime::Context*>(context);
    delete x;
  }
}

NNADAPTER_EXPORT int NNAdapterModel_create(NNAdapterModel** model) {
  if (!model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = new nnadapter::runtime::Model();
  if (m == nullptr) {
    *model = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *model = reinterpret_cast<NNAdapterModel*>(m);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterModel_destroy(NNAdapterModel* model) {
  if (model) {
    auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
    delete m;
  }
}

NNADAPTER_EXPORT int NNAdapterModel_finish(NNAdapterModel* model) {
  if (!model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  return m->Finish();
}

NNADAPTER_EXPORT int NNAdapterModel_addOperand(NNAdapterModel* model,
                                               const NNAdapterOperandType* type,
                                               NNAdapterOperand** operand) {
  if (!model || !type || !operand) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  nnadapter::core::Operand* o = nullptr;
  int result = m->AddOperand(*type, &o);
  if (result == NNADAPTER_NO_ERROR) {
    *operand = reinterpret_cast<NNAdapterOperand*>(o);
  }
  return result;
}

NNADAPTER_EXPORT int NNAdapterModel_setOperandValue(NNAdapterOperand* operand,
                                                    void* buffer,
                                                    uint32_t length,
                                                    bool copy) {
  if (!operand || !buffer || !length) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto o = reinterpret_cast<nnadapter::core::Operand*>(operand);
  if (copy) {
    o->buffer = malloc(length);
    if (!o->buffer) return NNADAPTER_OUT_OF_MEMORY;
    memcpy(o->buffer, buffer, length);
    o->type.lifetime = NNADAPTER_CONSTANT_COPY;
  } else {
    o->buffer = buffer;
    o->type.lifetime = NNADAPTER_CONSTANT_REFERENCE;
  }
  o->length = length;
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterModel_getOperandType(
    NNAdapterOperand* operand, NNAdapterOperandType** type) {
  if (!operand || !type) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto o = reinterpret_cast<nnadapter::core::Operand*>(operand);
  *type = &o->type;
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterModel_addOperation(
    NNAdapterModel* model,
    NNAdapterOperationType type,
    uint32_t input_count,
    NNAdapterOperand** input_operands,
    uint32_t output_count,
    NNAdapterOperand** output_operands,
    NNAdapterOperation** operation) {
  if (!model || !operation || !output_count || !output_operands) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  nnadapter::core::Operation* o = nullptr;
  int result = m->AddOperation(type, &o);
  if (result != NNADAPTER_NO_ERROR) {
    return result;
  }
  o->input_operands.resize(input_count);
  for (uint32_t i = 0; i < input_count; i++) {
    o->input_operands[i] =
        reinterpret_cast<nnadapter::core::Operand*>(input_operands[i]);
  }
  o->output_operands.resize(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    o->output_operands[i] =
        reinterpret_cast<nnadapter::core::Operand*>(output_operands[i]);
  }
  switch (type) {
#define REGISTER_OPERATION(__op_type__,             \
                           __validate_func_name__,  \
                           __prepare_func_name__,   \
                           __execute_func_name__,   \
                           ...)                     \
  case NNADAPTER_##__op_type__:                     \
    nnadapter::operation::__prepare_func_name__(o); \
    break;
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_OPERATION_ALL_H__
#undef REGISTER_OPERATION
    default:
      NNADAPTER_LOG(WARNING) << "Unsupported operation("
                             << nnadapter::OperationTypeToString(type)
                             << ") is found.";
      break;
  }
  *operation = reinterpret_cast<NNAdapterOperation*>(o);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int NNAdapterModel_identifyInputsAndOutputs(
    NNAdapterModel* model,
    uint32_t input_count,
    NNAdapterOperand** input_operands,
    uint32_t output_count,
    NNAdapterOperand** output_operands) {
  if (!model || !output_operands || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  auto is = reinterpret_cast<nnadapter::core::Operand**>(input_operands);
  auto os = reinterpret_cast<nnadapter::core::Operand**>(output_operands);
  return m->IdentifyInputsAndOutputs(input_count, is, output_count, os);
}

NNADAPTER_EXPORT int NNAdapterModel_getSupportedOperations(
    const NNAdapterModel* model,
    NNAdapterContext* context,
    bool* supported_operations) {
  if (!model || !context || !supported_operations) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<const nnadapter::runtime::Model*>(model);
  auto x = reinterpret_cast<nnadapter::runtime::Context*>(context);
  return m->GetSupportedOperations(x, supported_operations);
}

NNADAPTER_EXPORT int NNAdapterCompilation_create(
    NNAdapterModel* model,
    const char* cache_token,
    void* cache_buffer,
    uint32_t cache_length,
    const char* cache_dir,
    NNAdapterContext* context,
    NNAdapterCompilation** compilation) {
  if (!context || !compilation) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<nnadapter::runtime::Model*>(model);
  auto x = reinterpret_cast<nnadapter::runtime::Context*>(context);
  auto c = new nnadapter::runtime::Compilation(
      m, cache_token, cache_buffer, cache_length, cache_dir, x);
  if (c == nullptr) {
    *compilation = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *compilation = reinterpret_cast<NNAdapterCompilation*>(c);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterCompilation_destroy(
    NNAdapterCompilation* compilation) {
  if (compilation) {
    auto c = reinterpret_cast<nnadapter::runtime::Compilation*>(compilation);
    delete c;
  }
}

NNADAPTER_EXPORT int NNAdapterCompilation_finish(
    NNAdapterCompilation* compilation) {
  if (!compilation) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = reinterpret_cast<nnadapter::runtime::Compilation*>(compilation);
  return c->Finish();
}

NNADAPTER_EXPORT int NNAdapterCompilation_queryInputsAndOutputs(
    NNAdapterCompilation* compilation,
    uint32_t* input_count,
    NNAdapterOperandType** input_types,
    uint32_t* output_count,
    NNAdapterOperandType** output_types) {
  if (!compilation || !input_count || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = reinterpret_cast<nnadapter::runtime::Compilation*>(compilation);
  return c->QueryInputsAndOutputs(
      input_count, input_types, output_count, output_types);
}

NNADAPTER_EXPORT int NNAdapterExecution_create(
    NNAdapterCompilation* compilation, NNAdapterExecution** execution) {
  if (!compilation || !execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = reinterpret_cast<nnadapter::runtime::Compilation*>(compilation);
  auto e = new nnadapter::runtime::Execution(c);
  if (e == nullptr) {
    *execution = nullptr;
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *execution = reinterpret_cast<NNAdapterExecution*>(e);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT void NNAdapterExecution_destroy(
    NNAdapterExecution* execution) {
  if (execution) {
    auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
    delete e;
  }
}

NNADAPTER_EXPORT int NNAdapterExecution_setInput(
    NNAdapterExecution* execution,
    int32_t index,
    void* memory,
    void* (*access)(void* memory,
                    NNAdapterOperandType* type,
                    void* device_buffer)) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  return e->SetInput(index, memory, access);
}

NNADAPTER_EXPORT int NNAdapterExecution_setOutput(
    NNAdapterExecution* execution,
    int32_t index,
    void* memory,
    void* (*access)(void* memory,
                    NNAdapterOperandType* type,
                    void* device_buffer)) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  return e->SetOutput(index, memory, access);
}

NNADAPTER_EXPORT int NNAdapterExecution_compute(NNAdapterExecution* execution) {
  if (!execution) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto e = reinterpret_cast<nnadapter::runtime::Execution*>(execution);
  return e->Compute();
}

#ifdef __cplusplus
}
#endif
