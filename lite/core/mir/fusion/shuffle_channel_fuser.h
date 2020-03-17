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

class ShuffleChannelFuser : public FuseBase {
 public:
  explicit ShuffleChannelFuser(const std::string& reshape_type,
                               const std::string& transpose_type,
                               const std::string& sub_structure)
      : reshape_type_(reshape_type),
        transpose_type_(transpose_type),
        sub_structure_(sub_structure) {}

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  // reshape or reshape2
  std::string reshape_type_;
  // transpose or transpose2
  std::string transpose_type_;
  // r_t_r or t_r
  // r_t_r: reshape + transpose + reshape
  // s_t_r: stack + transpose + reshape
  std::string sub_structure_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
