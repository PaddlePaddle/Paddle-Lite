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

#include <map>
#include <vector>
#include "core/hal/types.h"

namespace nnadapter {

class NCHW2NHWCDataLayoutConverter {
 public:
  void Apply(hal::Model* model);
  void SetPermutation(hal::Operand* operand,
                      const std::vector<int32_t>& permutation);
  std::vector<int32_t> GetPermutation(hal::Operand* operand);
  void SetOperationLayout(hal::Operation* operation,
                          const int input_num = 1,
                          const int output_num = 1);
  hal::Model* GetModel();
  virtual void ConvertConv2D(hal::Operation* operation);
  virtual ~NCHW2NHWCDataLayoutConverter() = default;

 private:
  // Operation converters
  void ConvertAdaptivePool2D(hal::Operation* operation);
  void ConvertCast(hal::Operation* operation);
  void ConvertClip(hal::Operation* operation);
  void ConvertConv2DTranspose(hal::Operation* operation);
  void ConvertElementwise(hal::Operation* operation);
  void ConvertPool2D(hal::Operation* operation);
  void ConvertConcat(hal::Operation* operation);
  void ConvertFill(hal::Operation* operation);
  void ConvertFillLike(hal::Operation* operation);
  void ConvertFlatten(hal::Operation* operation);
  void ConvertFullyConnected(hal::Operation* operation);
  void ConvertGather(hal::Operation* operation);
  void ConvertLeakyRelu(hal::Operation* operation);
  void ConvertLpNormalization(hal::Operation* operation);
  void ConvertActivation(hal::Operation* operation);
  void ConvertPow(hal::Operation* operation);
  void ConvertQuantize(hal::Operation* operation);
  void ConvertReduce(hal::Operation* operation);
  void ConvertReshape(hal::Operation* operation);
  void ConvertResizeNearest(hal::Operation* operation);
  void ConvertResizeLinear(hal::Operation* operation);
  void ConvertShape(hal::Operation* operation);
  void ConvertSoftmax(hal::Operation* operation);
  void ConvertSplit(hal::Operation* operation);
  void ConvertSqueeze(hal::Operation* operation);
  void ConvertTranspose(hal::Operation* operation);
  void ConvertMatMul(hal::Operation* operation);

 private:
  hal::Model* model_{nullptr};
  std::map<hal::Operand*, std::vector<int32_t>> permutations_;
};

void ConvertDataLayoutNCHWToNHWC(hal::Model* model);

}  // namespace nnadapter
