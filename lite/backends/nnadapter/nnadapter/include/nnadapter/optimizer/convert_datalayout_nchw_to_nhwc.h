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
#include "core/types.h"

namespace nnadapter {

class NCHW2NHWCDataLayoutConverter {
 public:
  virtual void Apply(core::Model* model);
  virtual void ConvertOperations(std::vector<core::Operation*> operations);
  void SetPermutation(core::Operand* operand,
                      const std::vector<int32_t>& permutation);
  std::vector<int32_t> GetPermutation(core::Operand* operand);
  void SetOperationLayout(core::Operation* operation,
                          const int input_num = 1,
                          const int output_num = 1);
  core::Model* GetModel();
  virtual ~NCHW2NHWCDataLayoutConverter() = default;
  // Operation converters
  virtual void ConvertAdaptivePool2D(core::Operation* operation);
  virtual void ConvertBatchNormalization(core::Operation* operation);
  virtual void ConvertCast(core::Operation* operation);
  virtual void ConvertClip(core::Operation* operation);
  virtual void ConvertComparisons(core::Operation* operation);
  virtual void ConvertConv2D(core::Operation* operation);
  virtual void ConvertConv2DTranspose(core::Operation* operation);
  virtual void ConvertCumSum(core::Operation* operation);
  virtual void ConvertElementwise(core::Operation* operation);
  virtual void ConvertPool2D(core::Operation* operation);
  virtual void ConvertConcat(core::Operation* operation);
  virtual void ConvertFill(core::Operation* operation);
  virtual void ConvertFillLike(core::Operation* operation);
  virtual void ConvertFlatten(core::Operation* operation);
  virtual void ConvertFullyConnected(core::Operation* operation);
  virtual void ConvertGather(core::Operation* operation);
  virtual void ConvertGelu(core::Operation* operation);
  virtual void ConvertLayerNormalization(core::Operation* operation);
  virtual void ConvertLeakyRelu(core::Operation* operation);
  virtual void ConvertLpNormalization(core::Operation* operation);
  virtual void ConvertActivation(core::Operation* operation);
  virtual void ConvertPow(core::Operation* operation);
  virtual void ConvertQuantize(core::Operation* operation);
  virtual void ConvertReduce(core::Operation* operation);
  virtual void ConvertReshape(core::Operation* operation);
  virtual void ConvertResizeNearest(core::Operation* operation);
  virtual void ConvertResizeLinear(core::Operation* operation);
  virtual void ConvertShape(core::Operation* operation);
  virtual void ConvertSlice(core::Operation* operation);
  virtual void ConvertSoftmax(core::Operation* operation);
  virtual void ConvertSplit(core::Operation* operation);
  virtual void ConvertSqueeze(core::Operation* operation);
  virtual void ConvertStack(core::Operation* operation);
  virtual void ConvertTile(core::Operation* operation);
  virtual void ConvertTranspose(core::Operation* operation);
  virtual void ConvertMatMul(core::Operation* operation);
  virtual void ConvertUnsqueeze(core::Operation* operation);
  virtual void ConvertUnstack(core::Operation* operation);
  virtual void SkipConversion(core::Operation* operation);

 private:
  core::Model* model_{nullptr};
  std::map<core::Operand*, std::vector<int32_t>> permutations_;
};

void ConvertDataLayoutNCHWToNHWC(core::Model* model);

}  // namespace nnadapter
