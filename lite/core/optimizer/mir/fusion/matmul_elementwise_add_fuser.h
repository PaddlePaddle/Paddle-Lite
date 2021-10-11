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
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class MatmulElementwiseAddFuser : public FuseBase {
 public:
  explicit MatmulElementwiseAddFuser(bool with_relu,
                                     const std::unique_ptr<SSAGraph>& graph)
      : with_relu_(with_relu), graph_(graph) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  void CreatePattern();
  bool with_relu_;
  const std::unique_ptr<SSAGraph>& graph_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
