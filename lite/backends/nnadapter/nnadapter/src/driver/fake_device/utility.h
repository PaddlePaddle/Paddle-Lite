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
#include "fake_ddk/all.h"

namespace nnadapter {
namespace fake_device {

// Convert the NNAdapter types to the fake device types
fake_ddk::PrecisionType ConvertToFakeDevicePrecisionType(
    NNAdapterOperandPrecisionCode input_precision);
fake_ddk::DataLayoutType ConvertToFakeDeviceDataLayoutType(
    NNAdapterOperandLayoutCode input_layout);
std::vector<int32_t> ConvertToFakeDeviceDimensions(
    int32_t* input_dimensions, uint32_t input_dimensions_count);
fake_ddk::FuseType ConvertFuseCodeToFakeDeviceFuseType(int32_t fuse_code);

}  // namespace fake_device
}  // namespace nnadapter
