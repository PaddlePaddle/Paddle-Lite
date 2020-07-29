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

#include "lite/core/mir/fusion/conv_conv_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/mir/fusion/conv_conv_fuser.h"
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ConvConvFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // initialze fuser params
  std::vector<bool> conv_has_bias_cases{true, false};
  std::vector<std::string> conv_type_cases{"conv2d", "depthwise_conv2d"};
  bool has_fp32 = false;
  bool has_int8 = false;
  for (auto& place : graph->valid_places()) {
    if (place.target == TARGET(kARM)) {
      if (place.precision == PRECISION(kFloat)) {
        has_fp32 = true;
      }
      if (place.precision == PRECISION(kInt8)) {
        has_int8 = true;
      }
    } else {
      return;
    }
  }
  // only support arm-fp32
  if (has_int8 || (has_fp32 && has_int8)) {
    return;
  }
  // only support fp32 fusion
  for (auto conv_has_bias0 : conv_has_bias_cases) {
    for (auto conv_has_bias1 : conv_has_bias_cases) {
      for (auto conv_type0 : conv_type_cases) {
        for (auto conv_type1 : {"conv2d"}) {  // it mustbe 1x1s1p0_conv
          VLOG(4) << "conv_has_bias0:" << conv_has_bias0
                  << " conv_type0:" << conv_type0;
          VLOG(4) << "conv_has_bias1:" << conv_has_bias1
                  << " conv_type1:" << conv_type1;
          fusion::ConvConvFuser fuser(
              conv_type0, conv_type1, conv_has_bias0, conv_has_bias1);
          fuser(graph.get());
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_conv_conv_fuse_pass, paddle::lite::mir::ConvConvFusePass)
    .BindTargets({TARGET(kARM)});
