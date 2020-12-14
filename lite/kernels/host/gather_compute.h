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
#include <stdint.h>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename IndexType, typename AxisType>
class GatherCompute : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  void Run() override;

  ~GatherCompute() {}
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::host::GatherCompute<int32_t, int32_t>
    GatherInt32Int32;
typedef paddle::lite::kernels::host::GatherCompute<int64_t, int64_t>
    GatherInt64Int64;
typedef paddle::lite::kernels::host::GatherCompute<int64_t, int32_t>
    GatherInt64Int32;
typedef paddle::lite::kernels::host::GatherCompute<int32_t, int64_t>
    GatherInt32Int64;
