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

#include "driver/cambricon_mlu/optimizer/convert_datalayout_nchw_to_nhwc.h"
#include <map>
#include <vector>
#include "optimizer/convert_datalayout_nchw_to_nhwc.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

static const std::vector<int32_t> kNCHW2NHWC = {0, 2, 3, 1};
static const std::vector<int32_t> kNHWC2NCHW = {0, 3, 1, 2};

class NCHW2NHWCDataLayoutConverter
    : public nnadapter::NCHW2NHWCDataLayoutConverter {
 public:
  void ConvertConv2D(core::Operation* operation);
};

void NCHW2NHWCDataLayoutConverter::ConvertConv2D(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 9);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dimensions_count, 4);
  auto filter_operand = input_operands[1];
  bool is_per_channel =
      filter_operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto group = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  // Check depthwise mode
  auto input_channel_index = TransposeAxis(1 /* NCHW */, input_permutation);
  NNADAPTER_CHECK_GE(input_channel_index, 0);
  NNADAPTER_CHECK_LT(input_channel_index, input_dimensions_count);
  auto input_channel_size =
      input_operand->type.dimensions.data[input_channel_index];
  bool is_depthwise_mode = group != 1 && input_channel_size == group;
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        GetModel(), input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  std::vector<int32_t> filter_permutation = {};
  if (is_per_channel) {
    filter_operand->type.symm_per_channel_params.channel_dim =
        is_depthwise_mode ? 3 : 0;
  }
  if (is_depthwise_mode) {
    // [C_out, 1, filter_height, filter_width]->[1, filter_height, filter_width,
    // C_out]
    filter_permutation = {1, 2, 3, 0};
  } else {
    // MagicMind requires filter_layout is HWCN.
    filter_permutation = {2, 3, 1, 0};
  }
  TransposeOperand(filter_operand, filter_permutation);
  SetPermutation(filter_operand, filter_permutation);
  auto output_operand = output_operands[0];
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation, 3);
}

NNADAPTER_EXPORT void ConvertDataLayoutNCHWToNHWC(core::Model* model) {
  NCHW2NHWCDataLayoutConverter converter;
  converter.Apply(model);
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
