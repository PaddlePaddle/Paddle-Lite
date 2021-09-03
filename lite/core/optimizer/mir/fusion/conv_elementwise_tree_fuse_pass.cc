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

#include "lite/core/optimizer/mir/fusion/conv_elementwise_tree_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/conv_elementwise_tree_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ConvElementwiseTreeFusePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // initialze fuser params
  std::vector<bool> conv_has_prelu_alpha_cases{true, false};
  std::vector<bool> conv_has_bias_cases{true, false};
  // TODO(zhaoyang34): Support "depthwise_conv2d", "conv2d_transpose"
  std::vector<std::string> conv_type_cases{"conv2d"};
  // TODO(zhaoyang34): Support "elementwise_sub", "elementwise_mul",
  // "elementwise_div"
  std::vector<std::string> elementwise_type_cases{
      "elementwise_add", "fusion_elementwise_add_activation"};

  // start fuse using params
  for (auto conv_has_prelu_alpha : conv_has_prelu_alpha_cases) {
    for (auto conv_has_bias : conv_has_bias_cases) {
      for (auto conv_type : conv_type_cases) {
        for (auto elementwise_type : elementwise_type_cases) {
          VLOG(4) << " conv_type: " << conv_type
                  << "  conv_has_bias: " << conv_has_bias
                  << "  conv_has_prelu_alpha: " << conv_has_prelu_alpha
                  << "  elementwise_type: " << elementwise_type;
          fusion::ConvElementwiseTreeFuser fuser(
              conv_type, conv_has_bias, conv_has_prelu_alpha, elementwise_type);
          fuser.apply_impl(graph.get());
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_conv_elementwise_tree_fuse_pass,
                  paddle::lite::mir::ConvElementwiseTreeFusePass)
    .BindTargets({TARGET(kOpenCL)});
