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

//
// Created by chenyaohuang on 2021/12/17.
//

#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class Unsqueeze2Pad3dSqueeze2Fuser : public FuseBase {
 public:
  explicit Unsqueeze2Pad3dSqueeze2Fuser(const std::string& unsqueeze2_type,
                                        const std::string& pad3d_type,
                                        const std::string& squeeze2_type) {
    pad3d_type_ = pad3d_type;
    squeeze2_type_ = squeeze2_type;
    unsqueeze2_type_ = unsqueeze2_type;
  }

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string pad3d_type_{"pad3d"};
  std::string squeeze2_type_{"squeeze2"};
  std::string unsqueeze2_type_{"unsqueeze2"};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
