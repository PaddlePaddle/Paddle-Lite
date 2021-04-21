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

#pragma once

#include "lite/backends/nnadapter/nnadapter/nnadapter_types.h"

namespace paddle {
namespace lite {

class NNAdapter final {
 public:
  static NNAdapter& Global();

#define NNADAPTER_DECLARE_FUNCTION(name) name##_fn name;

  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_acquire);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_release);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getName);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVendor);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getType);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterDevice_getVersion);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_create);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_destroy);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_finish);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_addOperand);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_setOperand);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_addOperation);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_setOperation);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterGraph_identifyInputsAndOutputs);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_createFromGraph);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_createFromCache);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_destroy);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_finish);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterModel_setCaching);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_create);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_destroy);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setInput);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_setOutput);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterExecution_run);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterEvent_wait);
  NNADAPTER_DECLARE_FUNCTION(NNAdapterEvent_destroy);
#undef NNADAPTER_DECLARE_FUNCTION

 private:
  NNAdapter();
  NNAdapter(const NNAdapter&) = delete;
  NNAdapter& operator=(const NNAdapter&) = delete;
  bool Init();
  void* library_{nullptr};
};

}  // namespace lite
}  // namespace paddle
