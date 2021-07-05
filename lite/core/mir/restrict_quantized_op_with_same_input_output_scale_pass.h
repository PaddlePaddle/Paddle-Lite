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

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite_metal {
namespace mir {

// Due to some hardware limitations, we need to apply the constraints for the
// quantized ops(such as concat) that the inputs and outputs must have the same
// scale.
// Get the input and output scales of the target op nodes, figure out the new
// scale and update it to all related op nodes.
// Use the environment variable 'QUANT_INPUT_OUTPUT_SCALE_RESTRICT_METHOD' to
// specify how to caculate the new scale, includes the following methods:
// 0(default): The mean of the input and output scales.
// 1: The maximum of the input and output scales.
// 2: The minimum of the input and output scales.
//
// For example:
//     conv2d             conv2d              conv2d
// (out: scale_x)       (out: scale_y)    (out: scale_z)
//       |                   |                  |
//       |                   y                  |
//       |                   |                  |
//       |                   |                  |
//       x -------------- concat ---------------z
//       (in: scale_x,scale_y,scale_z out: scale_w)
//                           |
//           conv2d          |         conv2d
//        (out:scale_m)      w     (out: scale_n)
//              |            |           |
//              m-------- concat --------n
//            (in: scale_m,scale_n out: scale_v)
//                           |
//                           v
//
// After the pass is applied:
//     conv2d             conv2d              conv2d
// (out: new_scale)   (out: new_scale)   (out: new_scale)
//       |                   |                  |
//       |                   y                  |
//       |                   |                  |
//       |                   |                  |
//       x -------------- concat -------------- z
//  (in: new_scale,new_scale,new_scale out: new_scale)
//                           |
//           conv2d          |         conv2d
//      (out:new_scale)      w    (out: new_scale)
//              |            |           |
//              m-------- concat --------n
//         (in: new_scale,new_scale out: new_scale)
//                           |
//                           v

class RestrictQuantizedOpWithSameInputOutputScalePass : public mir::StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
