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

#include "lite/core/mir/fusion/conv_activation_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/mir/fusion/conv_activation_fuser.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ConvActivationFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
    for (auto act_type : {"relu", "leaky_relu"}) {
      for (auto has_bias : {true, false}) {
        fusion::ConvActivationFuser fuser(conv_type, act_type, has_bias);
        fuser(graph.get());
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_conv_activation_fuse_pass,
                  paddle::lite::mir::ConvActivationFusePass)
    .BindTargets({TARGET(kAny)})
    .BindKernel("conv2d");
