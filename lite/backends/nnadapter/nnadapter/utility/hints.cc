// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "utility/hints.h"
#include "utility/logging.h"

namespace nnadapter {

void ClearTemporaryShapeInfo(void** handler) {
  NNADAPTER_CHECK(*handler);
  free(*handler);
  *handler = nullptr;
}

NNAdapterOperandDimensionType* GetTemporyShapeInfo(hal::Operand* operand) {
  return operand->hints[NNADAPTER_OPERAND_HINTS_KEY_TEMPORY_SHAPE].handler
             ? reinterpret_cast<NNAdapterOperandDimensionType*>(
                   operand->hints[NNADAPTER_OPERAND_HINTS_KEY_TEMPORY_SHAPE]
                       .handler)
             : nullptr;
}

void SetTemporyShapeInfo(hal::Operand* operand,
                         const NNAdapterOperandDimensionType type) {
  auto ptr = malloc(sizeof(NNAdapterOperandDimensionType));
  memcpy(ptr, &type, sizeof(NNAdapterOperandDimensionType));
  operand->hints[NNADAPTER_OPERAND_HINTS_KEY_TEMPORY_SHAPE].handler = ptr;
  operand->hints[NNADAPTER_OPERAND_HINTS_KEY_TEMPORY_SHAPE].deleter =
      ClearTemporaryShapeInfo;
}

}  // namespace nnadapter
