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
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"

namespace nnadapter {
namespace verisilicon_timvx {

#define TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION \
  "TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION"

// Convert NNAdapter types to tim-vx types
tim::vx::DataType ConvertToTimVXDataType(
    NNAdapterOperandPrecisionCode input_precision);
tim::vx::DataLayout ConvertToTimVXDataLayout(
    NNAdapterOperandLayoutCode input_layout);
tim::vx::ShapeType ConvertToTimVXShapeType(int32_t* input_dimensions,
                                           uint32_t input_dimensions_count);
tim::vx::TensorAttribute ConvertToTimVXTensorAttribute(
    NNAdapterOperandLifetimeCode input_lifetime);
std::vector<uint32_t> ConvertToTimVXPermutation(const int32_t* input_perm_data,
                                                size_t input_perm_count);
int32_t ConvertToTimVXAxis(int32_t axis, size_t dimension_count);

// Create tim-vx tensor base on NNAdapter types
std::shared_ptr<tim::vx::Tensor> CreateTimVXTensor(
    tim::vx::Graph* graph,
    tim::vx::ShapeType shape,
    tim::vx::DataType data_type,
    const float* quant_scale = nullptr,
    const int32_t* quant_zero_point = nullptr,
    uint32_t quant_scale_count = 0,
    uint32_t quant_channel_dim = 0,
    void* buffer = nullptr,
    tim::vx::DataLayout data_layout = tim::vx::DataLayout::WHCN,
    tim::vx::TensorAttribute tensor_attr = tim::vx::TensorAttribute::TRANSIENT);
std::shared_ptr<tim::vx::Tensor> CreateTimVXTensor(
    tim::vx::Graph* graph,
    const NNAdapterOperandType* type,
    void* buffer = nullptr,
    std::vector<int32_t> dimensions = {});
}  // namespace verisilicon_timvx
}  // namespace nnadapter
