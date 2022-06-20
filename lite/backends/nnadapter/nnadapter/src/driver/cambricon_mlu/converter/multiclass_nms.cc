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

#include "driver/cambricon_mlu/converter.h"
#include "operation/non_max_suppression.h"
#include "utility/debug.h"
#include "utility/logging.h"
namespace nnadapter {
namespace cambricon_mlu {

int ConvertNonMaxSuppression(Converter* converter, core::Operation* operation) {
  NON_MAX_SUPPRESSION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto box_tensor = converter->GetMappedTensor(bboxes_operand);
  if (!box_tensor) {
    box_tensor = converter->ConvertOperand(bboxes_operand);
  }
  auto score_tensor = converter->GetMappedTensor(scores_operand);
  if (!score_tensor) {
    score_tensor = converter->ConvertOperand(scores_operand);
  }

  auto multiclass_nms_node =
      converter->network()->AddIMulticlassNmsNode(box_tensor, score_tensor);
  NNADAPTER_CHECK(multiclass_nms_node)
      << "Failed to add non_max_suppression node.";
  multiclass_nms_node->SetNumEntry(5);
  multiclass_nms_node->SetBackGroundLabelVal(
      static_cast<int64_t>(background_label));
  multiclass_nms_node->SetScoreThresholdVal(score_threshold);
  multiclass_nms_node->SetNmsTopKVal(static_cast<int64_t>(nms_top_k));
  multiclass_nms_node->SetNmsThresholdVal(nms_threshold);
  multiclass_nms_node->SetKeepTopKVal(static_cast<int64_t>(keep_top_k));

  auto out_tensor = multiclass_nms_node->GetOutput(0);
  auto out_size_tensor = multiclass_nms_node->GetOutput(1);

  auto input_dims_ptr = input_operands[0]->type.dimensions.data;
  auto N = input_dims_ptr[0];

  int start_val = 0;
  auto start_tensor = converter->AddInt32ConstantTensor(&start_val, {1});
  int delta_val = 1;
  auto delta_tensor = converter->AddInt32ConstantTensor(&delta_val, {1});
  auto bs_tensor = converter->AddInt32ConstantTensor(&N, {1});

  auto range_node0 = converter->network()->AddIRangeNode(
      start_tensor, bs_tensor, delta_tensor);
  NNADAPTER_CHECK(range_node0) << "Failed to add range node.";

  auto logic_eq_node = converter->network()->AddILogicNode(
      out_size_tensor, start_tensor, magicmind::ILogic::EQ);
  NNADAPTER_CHECK(logic_eq_node) << "Failed to add logic node.";
  auto cast_node = converter->network()->AddICastNode(
      logic_eq_node->GetOutput(0), magicmind::DataType::INT32);
  NNADAPTER_CHECK(cast_node) << "Failed to add cast node.";
  auto mul_node = converter->network()->AddIElementwiseNode(
      cast_node->GetOutput(0), delta_tensor, magicmind::IElementwise::MUL);
  NNADAPTER_CHECK(mul_node) << "Failed to add mul node.";

  auto bias_add_node = converter->network()->AddIElementwiseNode(
      out_size_tensor, mul_node->GetOutput(0), magicmind::IElementwise::ADD);
  NNADAPTER_CHECK(bias_add_node) << "Failed to add bias_add node.";
  auto split_node0 = converter->network()->AddISplitNode(
      bias_add_node->GetOutput(0), start_tensor, N);
  NNADAPTER_CHECK(split_node0) << "Failed to add split node.";
  auto split_node1 = converter->network()->AddISplitNode(
      range_node0->GetOutput(0), start_tensor, N);
  NNADAPTER_CHECK(split_node1) << "Failed to add split node.";

  std::vector<magicmind::ITensor*> range_vec;
  for (int i = 0; i < N; i++) {
    auto out_size_i = split_node0->GetOutput(i);
    auto range_node = converter->network()->AddIRangeNode(
        start_tensor, out_size_i, delta_tensor);
    NNADAPTER_CHECK(range_node) << "Failed to add range node.";
    auto unsqueeze_node = converter->network()->AddIUnsqueezeNode(
        range_node->GetOutput(0), delta_tensor);
    NNADAPTER_CHECK(unsqueeze_node) << "Failed to add unsqueeze node.";

    auto expand_as_node = converter->network()->AddIExpandAsNode(
        split_node1->GetOutput(i), unsqueeze_node->GetOutput(0));
    NNADAPTER_CHECK(expand_as_node) << "Failed to add expand node.";
    std::vector<magicmind::ITensor*> batch_vec = {expand_as_node->GetOutput(0),
                                                  unsqueeze_node->GetOutput(0)};
    auto concat_node =
        converter->network()->AddIConcatNode(delta_tensor, batch_vec);
    NNADAPTER_CHECK(concat_node) << "Failed to add concat node.";
    range_vec.push_back(concat_node->GetOutput(0));
  }
  auto concat_node =
      converter->network()->AddIConcatNode(start_tensor, range_vec);
  NNADAPTER_CHECK(concat_node) << "Failed to add concat node.";
  auto gather_nd_node = converter->network()->AddIGatherNdNode(
      out_tensor, concat_node->GetOutput(0));
  NNADAPTER_CHECK(gather_nd_node) << "Failed to add gather_nd node.";

  std::vector<int> split_v_size = {1, 6};
  auto split_v_tensor =
      converter->AddInt32ConstantTensor(split_v_size.data(), {2});
  auto split_v_node = converter->network()->AddISplitNode(
      gather_nd_node->GetOutput(0), split_v_tensor, delta_tensor, 2);
  NNADAPTER_CHECK(split_v_node) << "Failed to add split node.";
  converter->UpdateTensorMap(output_box_operand, split_v_node->GetOutput(1));
  converter->UpdateTensorMap(output_index_operand, split_v_node->GetOutput(0));
  converter->UpdateTensorMap(output_nms_rois_num_operand,
                             bias_add_node->GetOutput(0));
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
