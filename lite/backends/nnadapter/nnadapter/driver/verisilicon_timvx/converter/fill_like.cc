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

#include "core/operation/fill_like.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertFillLike(Converter* converter, hal::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  std::vector<int32_t> dimensions = {1};
  std::shared_ptr<tim::vx::Tensor> dummy_tesnor;
  if (IsUInt8AsymmPerLayerQuantType(input_operand->type.precision)) {
    uint8_t value = 0;
    const float quant_scale = 0;
    const int32_t zero_point = 128;
    dummy_tesnor = converter->AddConstantTensor(
        &value, dimensions, tim::vx::DataType::INT8, &quant_scale, &zero_point);
  } else {
    float value = 0;
    dummy_tesnor = converter->AddConstantTensor(
        &value, dimensions, tim::vx::DataType::FLOAT32, nullptr, nullptr);
  }
  auto fill_like_op =
      converter->graph()->CreateOperation<tim::vx::ops::Multiply>(0.0f);
  fill_like_op->BindInputs({input_tensor, dummy_tesnor});
  fill_like_op->BindOutput({output_tensor});

  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
