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
#include <random>
#include <string>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/operators/dropout_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = lite::fluid::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class DropoutCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::DropoutParam;
  void Run() override {
    auto& param = *param_.get_mutable<operators::DropoutParam>();
    const auto* x_data = param.x->template data<T>();
    auto* out_data = param.output->template mutable_data<T>();
    if (!param.is_test) {
      auto* mask_data = param.mask->template mutable_data<T>();
      size_t size = param.mask->dims().production();
      // Special case when dropout_prob is 1.0
      if (param.dropout_prob == 1.0f) {
        std::memset(out_data, 0, size * sizeof(T));   // NOLINT
        std::memset(mask_data, 0, size * sizeof(T));  // NOLINT
        return;
      }
      // std::minstd_rand engine;
      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed_data = 0;
      if (param.seed_tensor->template data<int>()) {
        seed_data = *(param.seed_tensor->template data<int>());
      } else {
        seed_data = param.fix_seed ? param.seed : 0;
      }
      std::minstd_rand engine;
      engine.seed(seed_data);
      std::uniform_real_distribution<float> dist(0, 1);

      for (size_t i = 0; i < size; ++i) {
        if (dist(engine) < param.dropout_prob) {
          mask_data[i] = 0;
          out_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (param.dropout_implementation == "upscale_in_train") {
            out_data[i] = x_data[i] / static_cast<T>(1.0f - param.dropout_prob);
          } else {
            out_data[i] = x_data[i];
          }
        }
      }
    } else {
      auto X = EigenMatrix<T>::Reshape(*param.x, 1);
      auto Y = EigenMatrix<T>::Reshape(*param.output, 1);
      if (param.dropout_implementation == "upscale_in_train") {
        Y.device(lite::fluid::EigenDeviceType<lite::TargetType::kX86>()) = X;
      } else {
        Y.device(lite::fluid::EigenDeviceType<lite::TargetType::kX86>()) =
            X * static_cast<T>(1.0f - param.dropout_prob);
      }
    }
  }

  virtual ~DropoutCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
