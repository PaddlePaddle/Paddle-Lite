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
#include "core/types.h"
#include "eznpu/eznpu_pub.h"

namespace nnadapter {
namespace eeasytech_npu {

// Convert NNAdapter types to eznpu types
eeasy::nn::PrecisionType ConvertToEznnPrecisionType(
    NNAdapterOperandPrecisionCode input_precision);
eeasy::nn::DataLayoutType ConvertToEznnDataLayoutType(
    NNAdapterOperandLayoutCode input_layout);
std::vector<int32_t> ConvertToEznnDimensions(int32_t* input_dimensions,
                                             uint32_t input_dimensions_count);

// Create eznpu tensor base on NNAdapter types
std::shared_ptr<eeasy::nn::Tensor> CreateEznnTensor(
    eeasy::nn::Graph* graph,
    const std::string& name,
    int32_t* dimensions_data,
    uint32_t dimensions_count,
    eeasy::nn::PrecisionType precision,
    const float* quant_scale = nullptr,
    const int32_t* zero_point = nullptr,
    void* buffer = nullptr,
    eeasy::nn::DataLayoutType layout = eeasy::nn::DataLayoutType::NCHW);
std::shared_ptr<eeasy::nn::Tensor> CreateEznnTensor(
    eeasy::nn::Graph* graph,
    const std::string& name,
    const NNAdapterOperandType* type,
    void* buffer = nullptr,
    std::vector<int32_t> dimensions = {});
}  // namespace eeasytech_npu
}  // namespace nnadapter
