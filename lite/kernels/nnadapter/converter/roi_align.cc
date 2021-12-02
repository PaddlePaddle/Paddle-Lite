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

int ConvertRoiAlign(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // ROIs operand
  auto rois_name = op->Input("ROIs").front();
  auto rois_scale_name = "ROIs0_scale";
  std::vector<float> rois_scales;
  if (op->HasInputScale(rois_scale_name, true)) {
    rois_scales = op->GetInputScale(rois_scale_name, true);
  }
  auto rois_operand =
      converter->AddInputOperand(scope, rois_name, {}, rois_scales);

  // Batch_indices operand
  auto rois_num_name = op->Input("RoisNum").front();
  auto batch_indices_operand = converter->AddInputOperand(scope, rois_num_name);

  // Output height opearnd
  auto pooled_height = op->GetAttr<int>("pooled_height");
  auto output_height_operand = converter->AddConstantOperand(pooled_height);

  // Output width operand
  auto pooled_width = op->GetAttr<int>("pooled_width");
  auto output_width_operand = converter->AddConstantOperand(pooled_width);

  // Sampling ratio operand
  auto sampling_ratio = op->GetAttr<int>("sampling_ratio");
  auto sampling_ratio_operand = converter->AddConstantOperand(sampling_ratio);

  // Spatial scale operand
  auto spatial_scale = op->GetAttr<float>("spatial_scale");
  auto spatial_scale_operand = converter->AddConstantOperand(spatial_scale);

  // Aligned operand
  // auto aligned = op->GetAttr<bool>("aligned");
  bool aligned = false;
  auto aligned_operand = converter->AddConstantOperand(aligned);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Roi_align operation
  converter->AddOperation(NNADAPTER_ROI_ALIGN,
                          {input_operand,
                           rois_operand,
                           batch_indices_operand,
                           output_height_operand,
                           output_width_operand,
                           sampling_ratio_operand,
                           spatial_scale_operand,
                           aligned_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
