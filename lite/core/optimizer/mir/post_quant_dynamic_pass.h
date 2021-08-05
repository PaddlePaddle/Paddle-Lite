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
 * Use post_quant_dynamic method to quantize the model.
 * In optimization stage, if the data type of weights is fp32, quantize the
 * weights to int8/16. So the size of the quantized weights is reduced 4x/2x.
 * In inference stage, the quantized weights are dequantized to fp32 and run
 * all ops to get output.
 */
class PostQuantDynamicPass : public ProgramPass {
 public:
  // The ops in quant_ops will be applied post_quant_dynamic.
  // Default, quant_ops = {"conv2d", "mul", "lookup_table"}
  static std::vector<std::string> quant_ops;
  // For the ops in quant_axis1_ops, the quantized axis is 1.
  // Default, quant_axis1_ops = {"mul", "lookup_table"}
  static const std::vector<std::string> quant_axis1_ops;

 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  void SetQuantType(lite_api::QuantType quant_type) {
    quant_type_ = quant_type;
  }

 private:
  lite_api::QuantType quant_type_{lite_api::QuantType::QUANT_INT16};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
