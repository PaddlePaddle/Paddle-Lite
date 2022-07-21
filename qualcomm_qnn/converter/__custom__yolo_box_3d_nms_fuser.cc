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

#include "driver/qualcomm_qnn/operation/__custom__yolo_box_3d_nms_fuser.h"
#include "driver/qualcomm_qnn/converter/converter.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertCustomYoloBox3dNmsFuser(Converter* converter,
                                   core::Operation* operation) {
  CUSTOM_YOLO_BOX_3D_NMS_FUSER_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input0_tensor = converter->GetMappedTensor(input_operands[0]);
  auto input1_tensor = converter->GetMappedTensor(input_operands[1]);
  auto input2_tensor = converter->GetMappedTensor(input_operands[2]);
  auto imgsize_tensor = converter->GetMappedTensor(input_operands[3]);
  auto out_box_tensor = converter->GetMappedTensor(output_operands[0]);
  auto out_nms_rois_num_tensor = converter->GetMappedTensor(output_operands[1]);
  auto out_index_tensor = converter->GetMappedTensor(output_operands[2]);
  auto out_location_tensor = converter->GetMappedTensor(output_operands[3]);
  auto out_dim_tensor = converter->GetMappedTensor(output_operands[4]);
  auto out_alpha_tensor = converter->GetMappedTensor(output_operands[5]);
  // attr
  auto anchors0_param = converter->GetParam("anchors0", anchors0);
  auto anchors1_param = converter->GetParam("anchors1", anchors1);
  auto anchors2_param = converter->GetParam("anchors2", anchors2);
  auto class_num_param = converter->GetParam("class_num", class_num);
  auto conf_thresh_param = converter->GetParam("conf_thresh", conf_thresh);
  auto downsample_ratio0_param =
      converter->GetParam("downsample_ratio0", downsample_ratio0);
  auto downsample_ratio1_param =
      converter->GetParam("downsample_ratio1", downsample_ratio1);
  auto downsample_ratio2_param =
      converter->GetParam("downsample_ratio2", downsample_ratio2);
  auto scale_param = converter->GetParam("scale_x_y", scale_x_y);
  auto background_label_param =
      converter->GetParam("background_label", background_label);
  auto score_threshold_param =
      converter->GetParam("score_threshold", score_threshold);
  auto nms_top_k_param = converter->GetParam("nms_top_k", nms_top_k);
  auto nms_threshold_param =
      converter->GetParam("nms_threshold", nms_threshold);
  auto nms_eta_param = converter->GetParam("nms_eta", nms_eta);
  auto keep_top_k_param = converter->GetParam("keep_top_k", keep_top_k);
  auto normalized_param = converter->GetParam("normalized", normalized);
  converter->AddNode(
      "CustomYoloBox3dNmsFuser",
      {input0_tensor, input1_tensor, input2_tensor, imgsize_tensor},
      {out_box_tensor,
       out_nms_rois_num_tensor,
       out_index_tensor,
       out_location_tensor,
       out_dim_tensor,
       out_alpha_tensor},
      {anchors0_param,
       anchors1_param,
       anchors2_param,
       class_num_param,
       conf_thresh_param,
       downsample_ratio0_param,
       downsample_ratio1_param,
       downsample_ratio2_param,
       scale_param,
       background_label_param,
       score_threshold_param,
       nms_top_k_param,
       nms_threshold_param,
       nms_eta_param,
       keep_top_k_param,
       normalized_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
