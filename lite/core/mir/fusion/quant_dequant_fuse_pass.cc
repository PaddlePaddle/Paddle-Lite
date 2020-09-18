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

#include "lite/core/mir/fusion/quant_dequant_fuse_pass.h"
#include <list>
#include <memory>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/mir/fusion/quant_dequant_op_fuser.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void QuantDequantFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // delete quant node
  std::vector<std::string> quant_op_types = {
      "fake_quantize_range_abs_max", "fake_quantize_moving_average_abs_max"};
  for (auto& op_type : quant_op_types) {
    fusion::DeleteQuantOpFuser fuser(op_type);
    fuser(graph.get());
  }

  // fuse quantized node and dequant node
  std::vector<std::string> quantized_op_types = {
      "conv2d", "depthwise_conv2d", "conv2d_transpose", "mul"};
  for (auto& op_type : quantized_op_types) {
    fusion::DequantOpFuser fuser(op_type);
    fuser(graph.get());
  }
  for (auto& op_type : quantized_op_types) {
    fusion::ChannelWiseDequantOpFuser fuser(op_type);
    fuser(graph.get());
  }

  // process quant_dequant_node
  std::vector<std::string> quant_dequant_op_types = {
      "fake_quantize_dequantize_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max"};
  for (auto& op_type : quant_dequant_op_types) {
    fusion::DeleteQuantDequantOpFuser dqd_fuser(op_type);
    dqd_fuser(graph.get());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_quant_dequant_fuse_pass,
                  paddle::lite::mir::QuantDequantFusePass)
    .BindTargets({TARGET(kAny)});
