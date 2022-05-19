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

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {
/*
 * Use fp16_attribute_pass method to set fp16_ops attirbute in model.
 * if op has is_weight, then add weight_name_fp16 attirbute;
 * Then running model, Accroding to weight_name_fp16 attirbute, op's weight
 * transform FP32 to FP16 percision type.
 */
class FP16AttributePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  std::vector<std::string> fp16_ops_{"conv2d",
                                     "depthwise_conv2d",
                                     "conv2d_transpose",
                                     "fc",
                                     "gru",
                                     "sequence_conv",
                                     "elementwise_add",
                                     "elementwise_mul",
                                     "elementwise_div",
                                     "elementwise_sub",
                                     "matmul",
                                     "mul",
                                     "matmul_v2",
                                     "prelu"};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
