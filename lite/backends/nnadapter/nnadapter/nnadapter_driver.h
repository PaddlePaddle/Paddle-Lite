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

typedef struct Network {
  std::vector<Operand> operands;
  std::vector<Operation> operations;
  std::vector<Operand *> inputs;
  std::vector<Operand *> outputs;
} Network;

typedef struct Driver {
  const char *name;
  const char *vendor;
  NNAdapterDeviceType type;
  int32_t version;
  int32_t (*createContext)(void **context);
  void (*destroyContext)(void *context);
  int32_t (*buildModel)(Network *network, void *context, void **model);
  int32_t (*excuteModel)(void *context, void *model);
} Driver;

int VerifyNetwork(Network *network);

}  // namespace driver
}  // namespace nnadapter
