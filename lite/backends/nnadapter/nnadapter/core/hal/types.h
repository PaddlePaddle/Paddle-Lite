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
#include <string>
#include <vector>
#include "nnadapter.h"  // NOLINT

namespace nnadapter {
namespace hal {

typedef struct Operand {
  NNAdapterOperandType type;
  void* buffer;
  uint32_t length;
} Operand;

typedef struct Argument {
  int index;
  uint32_t dimension_count;
  int32_t dimensions[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
  void* buffer;
  uint32_t length;
} Argument;

typedef struct Operation {
  NNAdapterOperationType type;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Operation;

typedef struct Cache {
  std::string cache_key;
  void* cache_buffer;
  uint32_t cache_length;
  std::string cache_dir;
  std::vector<NNAdapterOperandType> input_types;
  std::vector<NNAdapterOperandType> output_types;
} Cache;

typedef struct Model {
  std::list<Operand> operands;
  std::list<Operation> operations;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Model;

typedef struct Device {
  const char* name;
  const char* vendor;
  NNAdapterDeviceType type;
  int32_t version;
  int (*create_context)(void** context);
  void (*destroy_context)(void* context);
  int (*create_program)(void* context,
                        Model* model,
                        Cache* cache,
                        void** program);
  void (*destroy_program)(void* program);
  int (*execute_program)(void* program,
                         uint32_t input_count,
                         Argument* input_arguments,
                         uint32_t output_count,
                         Argument* output_arguments);
} Device;

}  // namespace hal
}  // namespace nnadapter
