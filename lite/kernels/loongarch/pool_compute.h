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

#include <Eigen/Core>
#include "lite/backends/loongarch/fluid/eigen.h"
#include "lite/backends/loongarch/math/math_function.h"
#include "lite/backends/loongarch/math/pooling.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

template <typename T>
class PoolCompute : public KernelLite<TARGET(kLoongArch), PRECISION(kFloat)> {
 public:
  using param_t = operators::PoolParam;
  void Run() override {
    auto& context = ctx_->As<LoongArchContext>();
    auto& param = *param_.get_mutable<param_t>();
    if (param.global_pooling) {
      for (size_t i = 0; i < param.ksize.size(); ++i) {
        param.ksize[i] = static_cast<int>(param.x->dims()[i + 2]);
      }
    }
    switch (param.ksize.size()) {
      case 2: {
        if (param.pooling_type == "max") {
          paddle::lite::loongarch::math::Pool2dFunctor<
              lite::TargetType::kLoongArch,
              paddle::lite::loongarch::math::MaxPool<T>,
              T>
              pool2d_forward;
          paddle::lite::loongarch::math::MaxPool<T> pool_process;
          pool2d_forward(context,
                         param.x,
                         param.ksize,
                         param.strides,
                         *param.paddings,
                         pool_process,
                         true,
                         false,
                         param.output);
        } else if (param.pooling_type == "avg") {
          paddle::lite::loongarch::math::Pool2dFunctor<
              lite::TargetType::kLoongArch,
              paddle::lite::loongarch::math::AvgPool<T>,
              T>
              pool2d_forward;
          paddle::lite::loongarch::math::AvgPool<T> pool_process;
          pool2d_forward(context,
                         param.x,
                         param.ksize,
                         param.strides,
                         *param.paddings,
                         pool_process,
                         param.exclusive,
                         param.adaptive,
                         param.output);
        }
      } break;
      case 3: {
      } break;
    }
  }
  virtual ~PoolCompute() = default;
};

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
