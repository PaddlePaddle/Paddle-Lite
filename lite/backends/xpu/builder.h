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

#include <xtcl/xtcl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace xpu {

std::string UniqueName(const std::string& prefix);

// Build IR graph to model, and store model data into lite tensor
bool BuildModel(std::vector<xtcl::xExpr>& inputs,   // NOLINT
                std::vector<xtcl::xExpr>& outputs,  // NOLINT
                lite::Tensor* model_data);

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
