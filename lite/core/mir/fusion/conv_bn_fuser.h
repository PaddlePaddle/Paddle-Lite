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

#include <cmath>
#include <memory>
#include <string>
#include "lite/core/mir/pattern_matcher_high_api.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class ConvBNFuser : public FuseBase {
 public:
  explicit ConvBNFuser(const std::string& conv_type, const bool conv_has_bias)
      : conv_type_(conv_type), conv_has_bias_(conv_has_bias) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  void ComputeAlphaAndBeta(float* scale_d,
                           float* mean_d,
                           float* var_d,
                           float* alpha,
                           float* beta,
                           float eps,
                           int h,
                           int w) {
    for (int i = 0; i < h; i++) {
      alpha[i] = scale_d[i] / std::sqrt(var_d[i] + eps);
    }
    for (int i = 0; i < h; i++) {
      beta[i] = (-mean_d[i]) * alpha[i];
    }
  }

 private:
  std::string conv_type_{"conv2d"};
  bool conv_has_bias_{false};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
