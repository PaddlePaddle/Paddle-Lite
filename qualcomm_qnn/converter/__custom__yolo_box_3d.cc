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

#include "operation/__custom__yolo_box_3d.h"
#include "driver/qualcomm_qnn/converter/converter.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertCustomYoloBox3d(Converter* converter, core::Operation* operation) {
  CUSTOM_YOLO_BOX_3D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operands[0]);
  auto imgsize_tensor = converter->GetMappedTensor(input_operands[1]);
  auto boxes_tensor = converter->GetMappedTensor(output_operands[0]);
  auto score_tensor = converter->GetMappedTensor(output_operands[1]);
  auto location_tensor = converter->GetMappedTensor(output_operands[2]);
  auto dim_tensor = converter->GetMappedTensor(output_operands[3]);
  auto alpha_tensor = converter->GetMappedTensor(output_operands[4]);
  // attr
  auto anchors_param = converter->GetParam(
      "anchors",
      std::vector<uint32_t>(anchors.data(), anchors.data() + anchors_count));
  auto class_num_param =
      converter->GetParam("class_num", static_cast<uint32_t>(class_num));
  auto conf_thresh_param =
      converter->GetParam("conf_thresh", static_cast<float>(conf_thresh));
  auto downsample_ratio_param = converter->GetParam(
      "downsample_ratio", static_cast<uint32_t>(downsample_ratio));
  auto scale_param =
      converter->GetParam("scale_x_y", static_cast<float>(scale_x_y));

  converter->AddNode(
      "CustomYoloBox3d",
      {input_tensor, imgsize_tensor},
      {boxes_tensor, score_tensor, location_tensor, dim_tensor, alpha_tensor},
      {anchors_param,
       class_num_param,
       conf_thresh_param,
       downsample_ratio_param,
       scale_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
