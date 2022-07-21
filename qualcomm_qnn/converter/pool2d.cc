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

#include "operation/pool2d.h"
#include "driver/qualcomm_qnn/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  auto output_tensor = converter->GetMappedTensor(output_operand);
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    auto pad_param = converter->GetParam(
        QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT,
        std::vector<uint32_t>{static_cast<uint32_t>(pad_height_top),
                              static_cast<uint32_t>(pad_height_bottom),
                              static_cast<uint32_t>(pad_width_left),
                              static_cast<uint32_t>(pad_width_right)},
        {2, 2});
    auto filter_param = converter->GetParam(
        QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
        std::vector<uint32_t>{static_cast<uint32_t>(kernel_height),
                              static_cast<uint32_t>(kernel_width)});
    auto stride_param = converter->GetParam(
        QNN_OP_POOL_AVG_2D_PARAM_STRIDE,
        std::vector<uint32_t>{static_cast<uint32_t>(stride_height),
                              static_cast<uint32_t>(stride_width)});
    auto rounding_mode_param =
        converter->GetParam(QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE,
                            static_cast<uint32_t>(ceil_mode));
    auto count_pad_param =
        converter->GetParam(QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES, flag);
    converter->AddNode(QNN_OP_POOL_AVG_2D,
                       {input_tensor},
                       {output_tensor},
                       {pad_param,
                        filter_param,
                        stride_param,
                        rounding_mode_param,
                        count_pad_param});
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    auto pad_param = converter->GetParam(
        QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT,
        std::vector<uint32_t>{static_cast<uint32_t>(pad_height_top),
                              static_cast<uint32_t>(pad_height_bottom),
                              static_cast<uint32_t>(pad_width_left),
                              static_cast<uint32_t>(pad_width_right)},
        {2, 2});
    auto filter_param = converter->GetParam(
        QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
        std::vector<uint32_t>{static_cast<uint32_t>(kernel_height),
                              static_cast<uint32_t>(kernel_width)});
    auto stride_param = converter->GetParam(
        QNN_OP_POOL_MAX_2D_PARAM_STRIDE,
        std::vector<uint32_t>{static_cast<uint32_t>(stride_height),
                              static_cast<uint32_t>(stride_width)});
    converter->AddNode(QNN_OP_POOL_MAX_2D,
                       {input_tensor},
                       {output_tensor},
                       {pad_param, filter_param, stride_param});
  } else {
    NNADAPTER_LOG(FATAL) << "Not support op: "
                         << OperationTypeToString(operation->type);
    return NNADAPTER_FEATURE_NOT_SUPPORTED;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
