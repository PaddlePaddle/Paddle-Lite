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

#include "lite/core/optimizer/mir/fusion/elementwise_add_activation_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/elementwise_add_activation_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ElementwiseActivationFusePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // initialze fuser params
  std::vector<std::string> elt_types{
      "elementwise_add", "elementwise_sub", "elementwise_mul"};
  std::vector<std::string> act_types{"relu"};

  auto has_target = [&](TargetType t) -> bool {
    for (auto& place : graph->valid_places()) {
      if (place.target == t) {
        return true;
      }
    }
    return false;
  };

  bool has_arm = has_target(TARGET(kARM));
  // arm not support tanh and abs act fusion
  if (!has_arm) {
    act_types.push_back("abs");
    act_types.push_back("tanh");
  }

  bool has_opencl = has_target(TARGET(kOpenCL));
  if (has_opencl) {
    act_types.push_back("relu6");
    act_types.push_back("gelu");
  }

  // start fuse using params
  for (auto elt_type : elt_types) {
    for (auto act_type : act_types) {
      fusion::ElementwiseActivationFuser fuser(elt_type, act_type);
      fuser(graph.get());
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_elementwise_activation_fuse_pass,
                  paddle::lite::mir::ElementwiseActivationFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU)})
    .ExcludeTargets({TARGET(kBM)})
    .ExcludeTargets({TARGET(kNNAdapter)})
    .BindKernel("fusion_elementwise_add_activation")
    .BindKernel("fusion_elementwise_sub_activation");
