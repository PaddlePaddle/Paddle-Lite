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

int ConvertMulticlassNms(Converter* converter, OpInfo* op, Scope* scope) {
  // bboxes operand
  auto bboxes_name = op->Input("BBoxes").front();
  auto bboxes_scale_name = "bboxes_scale";
  std::vector<float> bboxes_scales;
  if (op->HasInputScale(bboxes_scale_name, true)) {
    bboxes_scales = op->GetInputScale(bboxes_scale_name, true);
  }
  auto bboxes_operand =
      converter->AddInputOperand(scope, bboxes_name, {}, bboxes_scales);

  // scores operand
  auto scores_name = op->Input("Scores").front();
  auto scores_scale_name = "scores_scale";
  std::vector<float> scores_scales;
  if (op->HasInputScale(scores_scale_name, true)) {
    scores_scales = op->GetInputScale(scores_scale_name, true);
  }
  auto scores_operand =
      converter->AddInputOperand(scope, scores_name, {}, scores_scales);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);
  std::vector<std::string> output_arg_names = op->OutputArgumentNames();
  NNAdapterOperand* index_operand = nullptr;
  bool return_index = false;
  if (std::find(output_arg_names.begin(), output_arg_names.end(), "Index") !=
      output_arg_names.end()) {
    auto index_name = op->Output("Index").front();
    index_operand = converter->AddOutputOperand(index_name);
    return_index = true;
  }
  NNAdapterOperand* nms_rois_name_operand = nullptr;
  if (std::find(output_arg_names.begin(),
                output_arg_names.end(),
                "NmsRoisNum") != output_arg_names.end()) {
    auto nms_rois_name = op->Output("NmsRoisNum").front();
    nms_rois_name_operand = converter->AddOutputOperand(nms_rois_name);
  }

  // attrs operand
  int background_label = op->GetAttr<int>("background_label");
  auto background_label_operand =
      converter->AddConstantOperand(background_label);
  int keep_top_k = op->GetAttr<int>("keep_top_k");
  auto keep_top_k_operand = converter->AddConstantOperand(keep_top_k);
  int nms_top_k = op->GetAttr<int>("nms_top_k");
  auto nms_top_k_operand = converter->AddConstantOperand(nms_top_k);
  float score_threshold = op->GetAttr<float>("score_threshold");
  auto score_threshold_operand = converter->AddConstantOperand(score_threshold);
  float nms_threshold = op->GetAttr<float>("nms_threshold");
  auto nms_threshold_operand = converter->AddConstantOperand(nms_threshold);
  float nms_eta = op->GetAttr<float>("nms_eta");
  auto nms_eta_operand = converter->AddConstantOperand(nms_eta);
  bool normalized = false;
  if (op->HasAttr("normalized")) {
    normalized = op->GetAttr<bool>("normalized");
  }
  auto normalized_operand = converter->AddConstantOperand(normalized);
  auto return_index_operand = converter->AddConstantOperand(return_index);

  std::vector<std::string> input_arg_names = op->InputArgumentNames();
  NNAdapterOperand* rois_num_operand = nullptr;
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "RoisNum") !=
      input_arg_names.end()) {
    auto rois_num_name = op->Input("RoisNum");
    if (rois_num_name.size() > 0) {
      std::vector<float> rois_num_scales;
      if (op->HasInputScale(rois_num_name.front(), true)) {
        rois_num_scales = op->GetInputScale(rois_num_name.front(), true);
      }
      rois_num_operand = converter->AddInputOperand(
          scope, rois_num_name.front(), {}, rois_num_scales);
    }
  }

  // Assign operation
  std::vector<NNAdapterOperand*> input_operands = {bboxes_operand,
                                                   scores_operand,
                                                   rois_num_operand,
                                                   background_label_operand,
                                                   score_threshold_operand,
                                                   nms_top_k_operand,
                                                   nms_threshold_operand,
                                                   nms_eta_operand,
                                                   keep_top_k_operand,
                                                   normalized_operand,
                                                   return_index_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand,
                                                    nms_rois_name_operand};
  if (return_index) {
    output_operands.push_back(index_operand);
  }

  converter->AddOperation(
      NNADAPTER_NON_MAX_SUPPRESSION, &input_operands, &output_operands);

  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
