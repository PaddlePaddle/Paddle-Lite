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
  std::vector<std::string> act_types{"relu"};
  bool has_int8 = false;
  bool has_arm = false;
  bool has_opencl = false;
  bool has_cuda = false;
  bool has_x86 = false;
  for (auto& place : graph->valid_places()) {
    if (place.precision == PRECISION(kInt8)) {
      has_int8 = true;
    }
    if (place.target == TARGET(kARM)) {
      has_arm = true;
    }
    if (place.target == TARGET(kOpenCL)) {
      has_opencl = true;
    }
    if (place.target == TARGET(kCUDA)) {
      has_cuda = true;
    }
    if (place.target == TARGET(kX86)) {
      has_x86 = true;
    }
  }

  if (has_arm) {
    act_types.push_back("relu6");
    act_types.push_back("leaky_relu");
  }
  if (has_opencl) {
    act_types.push_back("relu6");
    act_types.push_back("leaky_relu");
    act_types.push_back("hard_swish");
    act_types.push_back("hard_sigmoid");
    act_types.push_back("prelu");
  }
  if (!has_int8 && has_cuda) {
    act_types.push_back("leaky_relu");
  }
  if (has_x86) {
    act_types.push_back("relu");
    act_types.push_back("relu6");
  }
  bool has_alpha = false;
  for (auto conv_type : {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
    for (auto act_type : act_types) {
      if (act_type == "prelu") {
        has_alpha = true;
      } else {
        has_alpha = false;
      }
      for (auto has_bias : {true, false}) {
        fusion::ConvActivationFuser fuser(
            conv_type, act_type, has_bias, has_alpha);
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
    .ExcludeTargets({TARGET(kXPU)})
    .ExcludeTargets({TARGET(kMLU)})
    .BindKernel("conv2d");
