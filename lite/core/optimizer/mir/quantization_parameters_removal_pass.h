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

#pragma once

#include <memory>
#include <set>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Remove the quantization parameters of some operators in the model according
 * to the configuration file, and force these operators to run on fp32/fp16
 * precision for the mixed precision inference.
 *
 *       w(int8)
 *         |
 *  in -> op(int8, with in_scale, out_scale and w_scale) -> out
 *
 * After applied:
 *
 *       w(fp32)
 *         |
 *  in -> op(fp32, without quant scale attrs) -> out
 *
 */
class QuantizationParametersRemovalPass : public mir::ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
