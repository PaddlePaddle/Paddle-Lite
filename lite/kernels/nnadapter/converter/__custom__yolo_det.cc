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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertCustomYoloDet(Converter* converter, OpInfo* op, Scope* scope) {
  // Inputs0 operand
  auto x0_name = op->Input("X0").front();
  auto x0_scale_name = "X0_scale";
  std::vector<float> x0_scales;
  if (op->HasInputScale(x0_scale_name, true)) {
    x0_scales = op->GetInputScale(x0_scale_name, true);
  }
  auto input0_operand =
      converter->AddInputOperand(scope, x0_name, {}, x0_scales);
  // Input1 operand
  auto x1_name = op->Input("X1").front();
  auto x1_scale_name = "X1_scale";
  std::vector<float> x1_scales;
  if (op->HasInputScale(x1_scale_name, true)) {
    x1_scales = op->GetInputScale(x1_scale_name, true);
  }
  auto input1_operand =
      converter->AddInputOperand(scope, x1_name, {}, x1_scales);
  // Input2 operand
  auto x2_name = op->Input("X2").front();
  auto x2_scale_name = "X2_scale";
  std::vector<float> x2_scales;
  if (op->HasInputScale(x2_scale_name, true)) {
    x2_scales = op->GetInputScale(x2_scale_name, true);
  }
  auto input2_operand =
      converter->AddInputOperand(scope, x2_name, {}, x2_scales);
  // ImgSize operand
  NNAdapterOperand* imgsize_operand = nullptr;
  auto imgsize_scale_name = "Y0_scale";
  auto imgsize_name = op->Input("ImgSize").front();
  std::vector<float> imgsize_scales;
  if (op->HasInputScale(imgsize_scale_name, true)) {
    imgsize_scales = op->GetInputScale(imgsize_scale_name, true);
  }
  imgsize_operand =
      converter->AddInputOperand(scope, imgsize_name, {}, imgsize_scales);

  auto anchors = op->GetAttr<std::vector<int>>("anchors");
  auto class_num = op->GetAttr<int>("class_num");
  auto conf_thresh = op->GetAttr<float>("conf_thresh");
  auto downsample_ratios = op->GetAttr<std::vector<int>>("downsample_ratios");
  auto nms_threshold = op->GetAttr<float>("nms_threshold");
  auto keep_top_k = op->GetAttr<int>("keep_top_k");

  auto anchors_operand = converter->AddConstantOperand(anchors);
  auto class_num_operand = converter->AddConstantOperand(class_num);
  auto conf_thresh_operand = converter->AddConstantOperand(conf_thresh);
  auto downsample_ratios_operand =
      converter->AddConstantOperand(downsample_ratios);
  auto nms_threshold_operand = converter->AddConstantOperand(nms_threshold);
  auto keep_top_k_operand = converter->AddConstantOperand(keep_top_k);

  // Output operand
  auto output_name = op->Output("Output").front();
  auto output_operand = converter->AddOutputOperand(output_name);

  converter->AddOperation(NNADAPTER_CUSTOM_YOLO_DET,
                          {input0_operand,
                           input1_operand,
                           input2_operand,
                           imgsize_operand,
                           anchors_operand,
                           class_num_operand,
                           conf_thresh_operand,
                           downsample_ratios_operand,
                           nms_threshold_operand,
                           keep_top_k_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
