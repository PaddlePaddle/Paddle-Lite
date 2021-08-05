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
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
Some op's inputs/outputs' precision is not correct due to unknown reasons.
For example: multiclass_nms2's output(Index) should be int32, but it is int64 in
specific models. We should update it to int32 by this pass.
*/
class FixMismatchedPrecisionPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  void FixMismatchedPrecision(
      const std::unique_ptr<SSAGraph>& graph,
      const std::string target_op_type,
      const std::string target_arg_name,
      const lite_api::PrecisionType target_precision_type);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
