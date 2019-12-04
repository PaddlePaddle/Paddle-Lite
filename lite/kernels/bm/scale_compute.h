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
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/operators/scale_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

class ScaleCompute : public KernelLite<TARGET(kBM), PRECISION(kFloat)> {
  public:
    using param_t = operators::ScaleParam;

    void PrepareForRun() override;
    void Run() override;

    virtual ~ScaleCompute() = default;
};

template <PrecisionType Ptype_out>
class ScaleComputeInt8 : public KernelLite<TARGET(kBM), PRECISION(kInt8)> {
  public:
    using param_t = operators::ScaleParam;
        
    void PrepareForRun() override;
    void Run() override;
        
    virtual ~ScaleComputeInt8() = default;
};
    

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
