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

#include "lite/core/optimizer/mir/fusion/conv_bn_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/conv_bn_fuser.h"
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ConvBNFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // initialze fuser params
  std::vector<bool> conv_has_bias_cases{true, false};
  std::vector<std::string> conv_type_cases{
      "conv2d", "depthwise_conv2d", "conv2d_transpose"};
  std::vector<std::string> bn_type_cases{"batch_norm", "sync_batch_norm"};
  // start fuse using params
  for (auto conv_has_bias : conv_has_bias_cases) {
    for (auto conv_type : conv_type_cases) {
      for (auto bn_type : bn_type_cases) {
        VLOG(4) << "conv_has_bias:" << conv_has_bias
                << " conv_type:" << conv_type;
        fusion::ConvBNFuser fuser(conv_type, bn_type, conv_has_bias);
        fuser(graph.get());
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_conv_bn_fuse_pass, paddle::lite::mir::ConvBNFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU), TARGET(kBM)});
