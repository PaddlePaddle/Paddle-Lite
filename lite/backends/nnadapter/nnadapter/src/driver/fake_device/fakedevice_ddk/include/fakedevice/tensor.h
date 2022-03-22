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
#include "fakedevice/types.h"

namespace fake_ddk {
namespace nn {

/* The structure of quantization parameter of Affine Asymmetric
 */
struct QuantizationParamAffineAsymmetric {
  std::vector<uint32_t> zero_point;
  std::vector<float> scale;
};

/* The structure of quantization parameter of Symmetric
 */
struct QuantizationParamSymmetric {
  std::vector<float> scale;
};

/* The structure of Tensor Attribute
 */
struct TensorAttr {
  std::string name;
  std::vector<uint32_t> dims;
  PrecisionType precision;
  DataLayoutType layout;
  /* Fake_ddk use affine asymmetric */
  QuantizationParamAffineAsymmetric qntParamAffineAsymmetric;
};

typedef struct _fakedevice_nn_tensor {
  /* Tensor attributes */
  std::shared_ptr<const TensorAttr> attr;
  void* data;
} fakedevice_nn_tensor_t;
typedef fakedevice_nn_tensor_t Tensor;
}  // namespace nn
}  // namespace fake_ddk
