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
#include "nnadapter.h"  // NOLINT

namespace nnadapter {
namespace core {

enum { NNADAPTER_MAX_SIZE_OF_HINTS = 8 };

typedef struct Hint {
  void* handler;
  void (*deleter)(void** handler);
} Hint;

typedef struct Operand {
  NNAdapterOperandType type;
  void* buffer;
  uint32_t length;
  Hint hints[NNADAPTER_MAX_SIZE_OF_HINTS];
} Operand;

typedef struct Argument {
  int index;
  void* memory;
  void* (*access)(void* memory,
                  NNAdapterOperandType* type,
                  void* device_buffer);
} Argument;

typedef struct Operation {
  NNAdapterOperationType type;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Operation;

typedef struct Cache {
  const char* token;
  const char* dir;
  std::vector<NNAdapterOperandType> input_types;
  std::vector<NNAdapterOperandType> output_types;
  std::vector<uint8_t> buffer;
} Cache;

typedef struct Model {
  std::list<Operand> operands;
  std::list<Operation> operations;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Model;

}  // namespace core
}  // namespace nnadapter
