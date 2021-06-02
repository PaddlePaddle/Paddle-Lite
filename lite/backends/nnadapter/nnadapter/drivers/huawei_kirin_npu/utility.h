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
#include "../../nnadapter_driver.h"   // NOLINT
#include "HiAiModelManagerService.h"  // NOLINT
#include "hiai_ir_build.h"            // NOLINT

namespace nnadapter {
namespace driver {
namespace huawei_kirin_npu {

std::shared_ptr<hiai::AiModelMngerClient> LoadOMModelFromBuffer(
    const std::string& model_name,
    std::vector<char>* model_buffer,
    bool* model_comp,
    int freq_level,
    int framework_type,
    int model_type,
    int device_type);
bool BuildOMModelToBuffer(std::vector<ge::Operator>& input_nodes,   // NOLINT
                          std::vector<ge::Operator>& output_nodes,  // NOLINT
                          std::vector<char>* model_buffer);

}  // namespace huawei_kirin_npu
}  // namespace driver
}  // namespace nnadapter
