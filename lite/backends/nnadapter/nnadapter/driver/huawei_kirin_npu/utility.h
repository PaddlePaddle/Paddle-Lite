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
#include "HiAiModelManagerService.h"  // NOLINT
#include "core/hal/types.h"
#include "hiai_ir_build.h"  // NOLINT

namespace nnadapter {
namespace huawei_kirin_npu {

std::shared_ptr<hiai::AiModelMngerClient> LoadOMModelFromBuffer(
    const std::string& model_name,
    std::vector<char>* model_buffer,
    bool* model_comp,
    int freq_level,
    int framework_type,
    int model_type,
    int device_type);
bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<char>* model_buffer);

ge::DataType ConvertPrecision(NNAdapterOperandPrecisionCode input_precision);
ge::Format ConvertDataLayout(NNAdapterOperandLayoutCode input_layout);
std::vector<int64_t> ConvertDimensions(int32_t* input_dimensions,
                                       uint32_t input_dimensions_count);
int32_t ConvertFuseCode(int32_t input_fuse_code);

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
