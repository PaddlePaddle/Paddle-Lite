// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

class TransformerAttentionFuser : public FuseBase {
 public:
  explicit TransformerAttentionFuser(bool reshape_has_xshape,
                                     bool transpose_has_xshape,
                                     bool dropout_mask,
                                     std::string mul_type)
      : reshape_has_xshape_(reshape_has_xshape),
        transpose_has_xshape_(transpose_has_xshape),
        dropout_mask_(dropout_mask),
        mul_type_(mul_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  bool reshape_has_xshape_;
  bool transpose_has_xshape_;
  bool dropout_mask_;
  std::string mul_type_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
