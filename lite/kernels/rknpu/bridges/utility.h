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

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

rk::nn::PrecisionType CvtPrecisionType(PrecisionType itype);

rk::nn::DataLayoutType CvtDataLayoutType(DataLayoutType itype);

std::vector<int32_t> CvtShape(const std::vector<int64_t>& in_shape);

std::vector<int32_t> CvtShape(const DDim& in_dims);

std::shared_ptr<rk::nn::Tensor> CvtTensor(
    rk::nn::Graph* graph,
    const std::string& name,
    const std::vector<int64_t>& shape,
    const std::vector<float>& scales,
    void* data = nullptr,
    PrecisionType precision = PRECISION(kInt8),
    DataLayoutType layout = DATALAYOUT(kNCHW));

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
