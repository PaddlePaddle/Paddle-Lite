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

#include "lite/core/optimizer/mir/fusion/scale_activation_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/scale_activation_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ScaleActivationFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (auto act_type : {"relu", "relu6", "leaky_relu"}) {
    fusion::ScaleActivationFuser fuser(act_type);
    fuser(graph.get());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_scale_activation_fuse_pass,
                  paddle::lite::mir::ScaleActivationFusePass)
    .BindTargets({TARGET(kARM)})
    .ExcludeTargets({TARGET(kNPU), TARGET(kXPU), TARGET(kNNAdapter)})
    .BindKernel("scale");
