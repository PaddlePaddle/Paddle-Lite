// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/operation/gather.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareGather(hal::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  auto& in_dims = input_operand->type.dimensions;
  auto& ids_dims = indices_operand->type.dimensions;
  auto& out_dims = output_operand->type.dimensions;
  int32_t in_count = in_dims.count;
  int32_t ids_count = ids_dims.count;

  auto infer_output_shape = [&](const int32_t* in_dims_data,
                                const int32_t* ids_dims_data,
                                int32_t* out_dims_data) {
    std::vector<int32_t> out_shape;
    for (int i = 0; i < axis; i++) {
      out_shape.push_back(in_dims_data[i]);
    }
    int32_t ids_numel = 1;
    for (int i = 0; i < ids_count; i++) {
      ids_numel *= ids_dims_data[i];
    }
    out_shape.push_back(ids_numel);
    for (int i = axis + 1; i < in_count; i++) {
      out_shape.push_back(in_dims_data[i]);
    }
    out_dims.count = out_shape.size();
    for (int i = 0; i < out_shape.size(); i++) {
      out_dims_data[i] = out_shape[i];
    }
  };

  infer_output_shape(in_dims.data, ids_dims.data, out_dims.data);
  for (uint32_t i = 0; i < in_dims.dynamic_count; i++) {
    infer_output_shape(in_dims.dynamic_data[i],
                       ids_dims.dynamic_data[i],
                       out_dims.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
