// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <set>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class ConvElementwiseTreeFuser : public FuseBase {
 public:
  explicit ConvElementwiseTreeFuser(const std::string& conv_type,
                                    const bool conv_has_bias,
                                    const bool conv_has_prelu_alpha,
                                    const std::string& elementwise_type) {
    conv_type_ = conv_type;
    conv_has_bias_ = conv_has_bias;
    conv_has_prelu_alpha_ = conv_has_prelu_alpha;
    elementwise_type_ = elementwise_type;
  }
  size_t apply_impl(SSAGraph* graph) {
    BuildPattern();
    PerformPatternMatcher(graph);

    for (const auto& matched : key2nodes_) {
      InsertNewNode(graph, matched);
    }

    GraphSafeRemoveNodes(graph, nodes2rm_);
    return key2nodes_.size();
  }

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

  std::string conv_type_{""};
  bool conv_has_bias_{false};
  bool conv_has_prelu_alpha_{false};
  std::string elementwise_type_{""};
  std::set<const Node*> nodes2rm_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
