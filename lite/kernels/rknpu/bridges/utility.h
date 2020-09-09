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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "rknpu/rknpu_pub.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

rk::nn::PrecisionType ToRknpuPrecisionType(PrecisionType precision);
rk::nn::DataLayoutType ToRknpuDataLayoutType(DataLayoutType layout);
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);
}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
