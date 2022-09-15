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
  void Apply(core::Model* model);
  void SetPermutation(core::Operand* operand,
                      const std::vector<int32_t>& permutation);
  std::vector<int32_t> GetPermutation(core::Operand* operand);
  void SetOperationLayout(core::Operation* operation,
                          const int input_num = 1,
                          const int output_num = 1);
  core::Model* GetModel();
  virtual void ConvertConv2D(core::Operation* operation);
  virtual void ConvertConv2DTranspose(core::Operation* operation);
  virtual ~NCHW2NHWCDataLayoutConverter() = default;

 private:
  // Operation converters
  void ConvertAdaptivePool2D(core::Operation* operation);
  void ConvertBatchNormalization(core::Operation* operation);
  void ConvertCast(core::Operation* operation);
  void ConvertChannelShuffle(core::Operation* operation);
  void ConvertClip(core::Operation* operation);
  void ConvertComparisons(core::Operation* operation);
  void ConvertCumSum(core::Operation* operation);
  void ConvertDequantize(core::Operation* operation);
  void ConvertElementwise(core::Operation* operation);
  void ConvertPool2D(core::Operation* operation);
  void ConvertConcat(core::Operation* operation);
  void ConvertFill(core::Operation* operation);
  void ConvertFillLike(core::Operation* operation);
  void ConvertFlatten(core::Operation* operation);
  void ConvertFullyConnected(core::Operation* operation);
  void ConvertGather(core::Operation* operation);
  void ConvertGelu(core::Operation* operation);
  void ConvertLayerNormalization(core::Operation* operation);
  void ConvertLeakyRelu(core::Operation* operation);
  void ConvertLpNormalization(core::Operation* operation);
  void ConvertActivation(core::Operation* operation);
  void ConvertPad(core::Operation* operation);
  void ConvertPow(core::Operation* operation);
  void ConvertQuantize(core::Operation* operation);
  void ConvertReduce(core::Operation* operation);
  void ConvertReshape(core::Operation* operation);
  void ConvertResizeNearest(core::Operation* operation);
  void ConvertResizeLinear(core::Operation* operation);
  void ConvertShape(core::Operation* operation);
  void ConvertSlice(core::Operation* operation);
  void ConvertSoftmax(core::Operation* operation);
  void ConvertSplit(core::Operation* operation);
  void ConvertSqueeze(core::Operation* operation);
  void ConvertStack(core::Operation* operation);
  void ConvertTile(core::Operation* operation);
  void ConvertTranspose(core::Operation* operation);
  void ConvertMatMul(core::Operation* operation);
  void ConvertUnsqueeze(core::Operation* operation);
  void ConvertUnstack(core::Operation* operation);

 private:
  core::Model* model_{nullptr};
  std::map<core::Operand*, std::vector<int32_t>> permutations_;
};

void ConvertDataLayoutNCHWToNHWC(core::Model* model);

}  // namespace nnadapter
