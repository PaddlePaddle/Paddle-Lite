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

#include "operation/roi_align.h"
#include <plugin_convert_rois.h>
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertRoiAlign(Converter* converter, core::Operation* operation) {
  ROI_ALIGN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto rois_tensor = converter->GetMappedTensor(rois_operand);
  if (!rois_tensor) {
    rois_tensor = converter->ConvertOperand(rois_operand);
  }
  auto batch_indices_tensor = converter->GetMappedTensor(batch_indices_operand);
  if (!batch_indices_tensor) {
    batch_indices_tensor = converter->ConvertOperand(batch_indices_operand);
  }
  magicmind::TensorMap plugin_inputs;
  std::vector<magicmind::ITensor*> rois{rois_tensor};
  std::vector<magicmind::ITensor*> rois_num{batch_indices_tensor};
  plugin_inputs.insert(
      std::pair<std::string, std::vector<magicmind::ITensor*>>("input", rois));
  plugin_inputs.insert(std::pair<std::string, std::vector<magicmind::ITensor*>>(
      "rois_num", rois_num));
  magicmind::DataTypeMap plugin_outputs_dtype;
  plugin_outputs_dtype["output"] = {magicmind::DataType::FLOAT16};
  auto convert_rois_node = converter->network()->AddIPluginNode(
      "PluginConvertRois", plugin_inputs, plugin_outputs_dtype);
  auto convert_rois_out_tensor = convert_rois_node->GetOutput(0);
  // Roi align
  auto roi_align_node = converter->network()->AddIRoiAlignNode(
      {input_tensor}, {convert_rois_out_tensor});
  NNADAPTER_CHECK(roi_align_node) << "Failed to add roi_align node.";
  magicmind::Layout input_layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  roi_align_node->SetLayout(input_layout, input_layout);
  roi_align_node->SetSamplingRatio(static_cast<int64_t>(sampling_ratio));
  roi_align_node->SetAligned(aligned);
  roi_align_node->SetSpatialScale(spatial_scale);
  roi_align_node->SetAlgo(magicmind::IRoiAlignAlgo::NONFPN);
  roi_align_node->SetRoiDefinition(magicmind::IRoiDefinition::BATCHID_CORNER);
  roi_align_node->SetPoolingMode(
      magicmind::IPoolingMode::AVERAGE_WITHOUT_PADDING);
  roi_align_node->SetOutputSize(static_cast<int64_t>(output_height),
                                static_cast<int64_t>(output_width));
  auto output_tensor = roi_align_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
