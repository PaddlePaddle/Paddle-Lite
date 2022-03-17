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
#include "fakedevice/fakedevice_pub.h"

namespace nnadapter {
namespace fake_device {

// Convert NNAdapter types to fakedevice types
fakedevice::nn::PrecisionType ConvertToFakedevicePrecisionType(
    NNAdapterOperandPrecisionCode input_precision);
fakedevice::nn::DataLayoutType ConvertToFakedeviceDataLayoutType(
    NNAdapterOperandLayoutCode input_layout);
std::vector<uint32_t> ConvertToFakedeviceDimensions(
    int32_t* input_dimensions, uint32_t input_dimensions_count);

// Create fakedevice tensor base on NNAdapter types
std::shared_ptr<fakedevice::nn::Tensor> CreateFakedeviceTensor(
    fakedevice::nn::Graph* graph,
    const std::string& name,
    int32_t* dimensions_data,
    uint32_t dimensions_count,
    fakedevice::nn::PrecisionType precision,
    const float* quant_scale = nullptr,
    const int32_t* zero_point = nullptr,
    void* buffer = nullptr,
    fakedevice::nn::DataLayoutType layout =
        fakedevice::nn::DataLayoutType::NCHW);
std::shared_ptr<fakedevice::nn::Tensor> CreateFakedeviceTensor(
    fakedevice::nn::Graph* graph,
    const std::string& name,
    const NNAdapterOperandType* type,
    void* buffer = nullptr,
    std::vector<int32_t> dimensions = {});

}  // namespace fake_device
}  // namespace nnadapter
