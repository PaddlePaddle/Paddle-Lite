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

#include "core/operation/slice.h"
#include <algorithm>
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

static void slice_offline_calc(const int32_t* input,
                               std::vector<int64_t> in_dims,
                               uint32_t axes_count,
                               int32_t* axes,
                               int32_t* starts,
                               int32_t* ends,
                               int32_t* out) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (size_t i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (size_t i = 0; i < axes_count; i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int length = in_dims.size();
  std::vector<int> dst_step(length);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(length);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

int PrepareSlice(hal::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* output_dimensions) {
    for (size_t i = 0; i < axes_count; ++i) {
      int dim = output_dimensions[axes[i]];
      if (dim > 0) {
        int start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
        int end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim);
        output_dimensions[axes[i]] = end - start;
      }
    }
  };

  infer_output_shape(output_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(output_operand->type.dimensions.dynamic_data[i]);
  }

  if (input_operand->type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
    auto tempory_shape_info =
        *(input_operand->hints[NNADAPTER_TEMPORY_SHAPE_INFO])
             .get_mutable<NNAdapterOperandDimensionType>();
    NNAdapterOperandDimensionType dimension_type;
    dimension_type.count = output_operand->type.dimensions.data[0];
    dimension_type.dynamic_count = input_operand->type.dimensions.dynamic_count;
    slice_offline_calc(tempory_shape_info.data,
                       std::vector<int64_t>({dimension_type.count}),
                       axes_count,
                       axes,
                       starts,
                       ends,
                       dimension_type.data);
    for (uint32_t i = 0; i < dimension_type.dynamic_count; i++) {
      slice_offline_calc(tempory_shape_info.dynamic_data[i],
                         std::vector<int64_t>({dimension_type.count}),
                         axes_count,
                         axes,
                         starts,
                         ends,
                         dimension_type.dynamic_data[i]);
    }
    output_operand->hints[NNADAPTER_TEMPORY_SHAPE_INFO].set(dimension_type);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
