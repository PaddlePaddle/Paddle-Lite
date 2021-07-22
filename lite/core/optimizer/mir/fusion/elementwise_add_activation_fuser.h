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

#include <memory>
#include <string>
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

// Detect elementwise and activation ops, and then merge into
// fusion_eltsiwise_act op.
// Example:
//  elementwise_add + relu fuse.
//    fusion::ElementwiseActivationFuser fuser("elementwise_add", "relu");
//    fuser(graph.get());
class ElementwiseActivationFuser : public FuseBase {
 public:
  explicit ElementwiseActivationFuser(const std::string& eltwise_type,
                                      const std::string& act_type)
      : eltwise_type_(eltwise_type), act_type_(act_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  std::string eltwise_type_;
  std::string act_type_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
