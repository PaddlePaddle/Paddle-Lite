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

#include "operation/pool2d.h"
#include "driver/rockchip_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace rockchip_npu {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to rknpu tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  rk::nn::PoolAttr attr;
  attr.ksize[0] = kernel_width;
  attr.ksize[1] = kernel_height;
  attr.stride[0] = stride_width;
  attr.stride[1] = stride_height;
  attr.pad[0] = pad_width_left;
  attr.pad[1] = pad_width_right;
  attr.pad[2] = pad_height_top;
  attr.pad[3] = pad_height_bottom;
  attr.pad_type = rk::nn::PadType::AUTO;
  // TODO(hong19860320) fix the order of kernel when global_pooling=true in
  // rknpu_ddk
  attr.global_pooling = false;
  attr.round_type = ceil_mode ? rk::nn::RoundType::ROUND_CEIL
                              : rk::nn::RoundType::ROUND_FLOOR;
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors = {input_tensor};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors = {output_tensor};
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    attr.pool_type = rk::nn::PoolType::POOLING_AVG;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    attr.pool_type = rk::nn::PoolType::POOLING_MAX;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  converter->AddOperator(
      rk::nn::OperatorType::POOL, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace rockchip_npu
}  // namespace nnadapter
