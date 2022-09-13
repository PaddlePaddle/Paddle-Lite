// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "optimizer/fuse_unsqueeze_pad_squeeze_into_pad.h"
#include <algorithm>
#include <map>
#include <vector>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class UnsqueezePadSqueezeFuser : public PatternMatcher {
 public:
  UnsqueezePadSqueezeFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void UnsqueezePadSqueezeFuser::BuildPattern() {
  // Unsqueeze
  auto unsqueeze_in = CreatePattern("unsqueeze_in")
                          ->IsOperationInputOperand(NNADAPTER_UNSQUEEZE, 0);
  auto unsqueeze_axes = CreatePattern("unsqueeze_axes")
                            ->IsOperationInputOperand(NNADAPTER_UNSQUEEZE, 1)
                            ->IsIntermediate();
  auto unsqueeze =
      CreatePattern("unsqueeze", NNADAPTER_UNSQUEEZE)->IsIntermediate();
  auto unsqueeze_out = CreatePattern("unsqueeze_out")
                           ->IsOperationOutputOperand(NNADAPTER_UNSQUEEZE, 0)
                           ->IsOperationInputOperand(NNADAPTER_PAD, 0)
                           ->CheckOutputCount(1)
                           ->IsIntermediate();
  // Pad
  auto pad_pads = CreatePattern("pad_pads")
                      ->IsOperationInputOperand(NNADAPTER_PAD, 1)
                      ->IsIntermediate();
  auto pad_mode =
      CreatePattern("pad_mode")->IsOperationInputOperand(NNADAPTER_PAD, 2);
  auto pad_value =
      CreatePattern("pad_value")->IsOperationInputOperand(NNADAPTER_PAD, 3);
  auto pad = CreatePattern("pad", NNADAPTER_PAD);
  auto pad_out = CreatePattern("pad_out")
                     ->IsOperationOutputOperand(NNADAPTER_PAD, 0)
                     ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 0)
                     ->CheckOutputCount(1)
                     ->IsIntermediate();
  // Squeeze
  auto squeeze_axes = CreatePattern("squeeze_axes")
                          ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 1)
                          ->IsIntermediate();
  auto squeeze = CreatePattern("squeeze", NNADAPTER_SQUEEZE)->IsIntermediate();
  auto squeeze_out = CreatePattern("squeeze_outout")
                         ->IsOperationOutputOperand(NNADAPTER_SQUEEZE, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> unsqueeze_ins{unsqueeze_in, unsqueeze_axes};
  unsqueeze_ins >> *unsqueeze >> *unsqueeze_out;
  std::vector<Pattern*> pad_ins{unsqueeze_out, pad_pads, pad_mode, pad_value};
  pad_ins >> *pad >> *pad_out;
  std::vector<Pattern*> squeeze_ins{pad_out, squeeze_axes};
  squeeze_ins >> *squeeze >> *squeeze_out;
}

bool UnsqueezePadSqueezeFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Match more patterns among nodes.
  auto unsqueeze_axes_operand = nodes.at("unsqueeze_axes")->operand;
  auto unsqueeze_axes_count = unsqueeze_axes_operand->length / sizeof(int32_t);
  auto unsqueeze_axes_data =
      reinterpret_cast<int32_t*>(unsqueeze_axes_operand->buffer);
  auto squeeze_axes_operand = nodes.at("squeeze_axes")->operand;
  auto squeeze_axes_count = squeeze_axes_operand->length / sizeof(int32_t);
  auto squeeze_axes_data =
      reinterpret_cast<int32_t*>(squeeze_axes_operand->buffer);
  if (unsqueeze_axes_count != 1 || squeeze_axes_count != 1 ||
      unsqueeze_axes_data[0] != squeeze_axes_data[0])
    return false;
  int axis = unsqueeze_axes_data[0];
  if (axis < 0) {
    axis += nodes.at("unsqueeze_in")->operand->type.dimensions.count + 1;
  }
  auto pad_pads_operand = nodes.at("pad_pads")->operand;
  auto pad_pads_count = pad_pads_operand->length / sizeof(int32_t);
  auto pad_pads_data = reinterpret_cast<int32_t*>(pad_pads_operand->buffer);
  if ((axis + 1) * 2 > pad_pads_count) return false;
  if (pad_pads_data[axis * 2] != 0 || pad_pads_data[axis * 2 + 1] != 0)
    return false;
  // Modify pad
  std::vector<int32_t> new_pads(pad_pads_count - 2);
  memcpy(new_pads.data(), pad_pads_data, sizeof(int32_t) * axis * 2);
  memcpy(new_pads.data() + axis * 2,
         pad_pads_data + axis * 2 + 2,
         sizeof(int32_t) * (pad_pads_count - 2 - axis * 2));
  auto new_pads_operand = AddInt32ConstantOperand(model, new_pads);
  auto& pad_inputs = nodes.at("pad")->operation->input_operands;
  auto& pad_outputs = nodes.at("pad")->operation->output_operands;
  pad_inputs[1] = new_pads_operand;
  pad_inputs[0] = nodes.at("unsqueeze_in")->operand;
  pad_outputs[0] = nodes.at("squeeze_outout")->operand;
  return true;
}

NNADAPTER_EXPORT void FuseUnsqueezePadSqueezeIntoPad(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply UnsqueezePadSqueezeFuser";
  bool stop;
  do {
    UnsqueezePadSqueezeFuser reshape_transpose_reshape_fuser;
    stop = reshape_transpose_reshape_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
