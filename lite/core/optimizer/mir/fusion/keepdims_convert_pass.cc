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

#include "lite/core/optimizer/mir/fusion/keepdims_convert_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/fusion/keepdims_convert_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void KeepdimsConvertPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  const std::vector<std::string> op_type_cases{"arg_max",
                                               "reduce_max",
                                               "reduce_min",
                                               "reduce_mean",
                                               "reduce_sum",
                                               "reduce_prob",
                                               "reduce_all",
                                               "reduce_any"};
  for (auto op_type : op_type_cases) {
    fusion::KeepdimsConvertFuser fuser(op_type);
    fuser(graph.get());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(keepdims_convert_pass, paddle::lite::mir::KeepdimsConvertPass)
    .BindTargets({TARGET(kOpenCL)});
