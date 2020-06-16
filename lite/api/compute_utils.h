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
#include <vector>
#include "paddle_api.h"  // NOLINT

namespace paddle {
namespace lite_api {

struct LITE_API ComputeUtils {
  static void TensorFloatToInt8(Tensor& tin,   // NOLINT
                                Tensor& tout,  // NOLINT
                                float scale);
  static void TensorFloatToInt8Inplace(Tensor& tin, float scale);  // NOLINT
  static void TensorInt8ToFloat(Tensor& tin,                       // NOLINT
                                Tensor& tout,                      // NOLINT
                                float scale);
  static void TensorInt8ToFloatInplace(Tensor& tin, float scale);  // NOLINT
  static void ConvWeightsFloatToInt8(Tensor& weightin,             // NOLINT
                                     Tensor& weightout,            // NOLINT
                                     std::vector<float> scale);
  static void ConvWeightsFloatToInt8Inplace(Tensor& weightin,  // NOLINT
                                            std::vector<float> scale);
  static void ConvWeightsInt8ToFloat(Tensor& weightin,   // NOLINT
                                     Tensor& weightout,  // NOLINT
                                     std::vector<float> scale);
  static void ConvWeightsInt8ToFloatInplace(Tensor& weightin,  // NOLINT
                                            std::vector<float> scale);
};

}  // namespace lite_api
}  // namespace paddle
