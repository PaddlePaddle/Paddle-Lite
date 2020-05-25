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

#include "lite/backends/x86/math/sequence_topk_avg_pooling.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SequenceTopkAvgPoolingCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceTopkAvgPoolingParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    lite::x86::math::SequenceTopkAvgPoolingFunctor<lite::TargetType::kX86, T>
        sequence_topk_avg_pooling;
    sequence_topk_avg_pooling(*param.X,
                              *param.ROW,
                              *param.COLUMN,
                              param.Out,
                              param.pos,
                              param.channel_num,
                              param.topks);
  };
  virtual ~SequenceTopkAvgPoolingCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
