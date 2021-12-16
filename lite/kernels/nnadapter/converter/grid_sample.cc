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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertGridSample(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Grid operand
  auto grid_name = op->Input("Grid").front();
  auto grid_scale_name = "Grid0_scale";
  std::vector<float> grid_scales;
  if (op->HasInputScale(grid_scale_name, true)) {
    grid_scales = op->GetInputScale(grid_scale_name, true);
  }
  auto grid_operand =
      converter->AddInputOperand(scope, grid_name, {}, grid_scales);

  // Align corners operand
  bool align_corners =
      op->HasAttr("align_corners") ? op->GetAttr<bool>("align_corners") : true;
  auto align_corners_operand = converter->AddConstantOperand(align_corners);

  // Mode opearnd
  std::string mode =
      op->HasAttr("mode") ? op->GetAttr<std::string>("mode") : "bilinear";
  NNAdapterInterpolateModeCode interpolate_code =
      ConvertInterpolateModeToNNInterpolateModeCode(mode);
  auto mode_operand = converter->AddConstantOperand(interpolate_code);

  // Pad mode operand
  std::string padding_mode = op->HasAttr("padding_mode")
                                 ? op->GetAttr<std::string>("padding_mode")
                                 : "zeros";
  NNAdapterPadModeCode pad_mode_code =
      ConvertPadModeToNNPadModeCode(padding_mode);
  auto pad_mode_operand = converter->AddConstantOperand(pad_mode_code);

  // Output operand
  auto output_name = op->Output("Output").front();
  auto output_operand = converter->AddOutputOperand(output_name);

  // Roi_align operation
  converter->AddOperation(NNADAPTER_GRID_SAMPLE,
                          {input_operand,
                           grid_operand,
                           align_corners_operand,
                           mode_operand,
                           pad_mode_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
