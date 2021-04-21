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

#pragma once

#include <list>
#include <vector>
#include "nnadapter_micros.h"  // NOLINT
#include "nnadapter_types.h"   // NOLINT

namespace nnadapter {
namespace driver {

typedef struct Operand {
  NNAdapterOperandType type;
  void *buffer;
  size_t length;
} Operand;

typedef struct Operation {
  NNAdapterOperationType type;
  std::vector<Operand *> inputs;
  std::vector<Operand *> outputs;
} Operation;

typedef struct Graph {
  std::list<Operand> operands;
  std::list<Operation> operations;
  std::vector<Operand *> inputs;
  std::vector<Operand *> outputs;
} Graph;

typedef struct Driver {
  const char *name;
  const char *vendor;
  NNAdapterDeviceType type;
  int32_t version;
  int (*createContext)(void **context);
  void (*destroyContext)(void *context);
  int (*createModelFromGraph)(void *context, Graph *graph, void **model);
  int (*createModelFromCache)(void *context,
                              void *buffer,
                              size_t length,
                              void **model);
  void (*destroyModel)(void *context, void *model);
  int (*runModelSync)(void *context,
                      void *model,
                      uint32_t inputCount,
                      Operand **inputs,
                      uint32_t outputCount,
                      Operand **outputs);
  // TODO(hong19860320) missing callback and notify defs
  int (*runModelAsync)(void *context,
                       void *model,
                       uint32_t inputCount,
                       Operand **inputs,
                       uint32_t outputCount,
                       Operand **outputs);
} Driver;

std::vector<Operation *> sortOperationsInTopologicalOrder(Graph *graph);

}  // namespace driver
}  // namespace nnadapter
