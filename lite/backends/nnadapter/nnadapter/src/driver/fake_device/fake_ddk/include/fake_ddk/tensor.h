// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>
#include "fake_ddk/types.h"

namespace fake_ddk {

/* The parameter of symmetric/asymmetric per-layer/per-channel quantization */
typedef struct {
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
  int32_t channel_dim;
} QuantParams;

/* Tensor Attribute */
typedef struct {
  PrecisionType precision;
  DataLayoutType layout;
  std::vector<int32_t> shape;
  QuantParams quant_params;
} TensorAttr;

/* Tensor */
typedef struct {
  TensorAttr attr;
  LifeTimeType lifetime;
  void* buffer;
  size_t length;
} Tensor;

}  // namespace fake_ddk
