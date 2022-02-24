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
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  tim::vx::PoolType pool_type;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    pool_type = tim::vx::PoolType::AVG;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    pool_type = tim::vx::PoolType::MAX;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  tim::vx::RoundType round_type =
      ceil_mode ? tim::vx::RoundType::CEILING : tim::vx::RoundType::FLOOR;
  auto pool2d_op = converter->graph()->CreateOperation<tim::vx::ops::Pool2d>(
      pool_type,
      std::array<uint32_t, 4>({static_cast<uint32_t>(pad_width_left),
                               static_cast<uint32_t>(pad_width_right),
                               static_cast<uint32_t>(pad_height_top),
                               static_cast<uint32_t>(pad_height_bottom)}),
      std::array<uint32_t, 2>({static_cast<uint32_t>(kernel_width),
                               static_cast<uint32_t>(kernel_height)}),
      std::array<uint32_t, 2>({static_cast<uint32_t>(stride_width),
                               static_cast<uint32_t>(stride_height)}),
      round_type);
  pool2d_op->BindInputs({input_tensor});
  pool2d_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
