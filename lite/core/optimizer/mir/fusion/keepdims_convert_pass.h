// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/optimizer/mir/pass.h"

/*
 * KeepdimsConvertPass splits some ops whose attribute `keepdims` or `keep_dim`
 * is false to two ops.
 * The reason for adding this pass is that it is hard for gpu to do reshape
 * opterations in arg_max/reduce_mean, etc,. So we split this problem.
 *
 * For example:
 *        |
 *       var1
 *        v
 *   OP: arg_max(keepdims=false)
 *        |
 *       var2
 *        v
 *
 * After this pass is applied:
 *        |
 *       var1
 *        v
 *   OP: arg_max(keepdims=true)
 *        |
 *    var2/trans
 *        v
 *   OP: reshape(shape = original arg_max's output dims)
 *        |
 *       var2
 *        v
 */

namespace paddle {
namespace lite {
namespace mir {

class KeepdimsConvertPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
