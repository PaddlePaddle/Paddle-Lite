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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertStridedSlice(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Axes
  auto axes = op->GetAttr<std::vector<int>>("axes");
  auto axes_size = static_cast<int>(axes.size());
  // Starts
  std::vector<int> starts;
  if (HasInput(op, scope, "StartsTensorList")) {
    LOG(FATAL) << "Not support StartsTensorList for stride_slice now.";
  } else if (HasInput(op, scope, "StartsTensor")) {
    LOG(FATAL) << "Not support StartsTensor for stride_slice now.";
  } else {
    starts = op->GetAttr<std::vector<int>>("starts");
  }
  // Ends
  std::vector<int> ends_ori;
  if (HasInput(op, scope, "EndsTensorList")) {
    LOG(FATAL) << "Not support EndsTensorList for stride_slice now.";
  } else if (HasInput(op, scope, "EndsTensor")) {
    LOG(FATAL) << "Not support EndsTensor for stride_slice now.";
  } else {
    ends_ori = op->GetAttr<std::vector<int>>("ends");
  }
  std::vector<int> ends(axes_size, 0);
  for (int i = 0; i < axes_size; i++) {
    auto dim = input_type->dimensions.data[axes[i]];
    CHECK(dim != NNADAPTER_UNKNOWN);
    ends[i] = ends_ori[i] > dim ? dim : ends_ori[i];
  }
  // Steps
  std::vector<int> steps;
  if (HasInput(op, scope, "StridesTensorList")) {
    LOG(FATAL) << "Not support StridesTensorList for stride_slice now.";
  } else if (HasInput(op, scope, "StridesTensor")) {
    LOG(FATAL) << "Not support StridesTensor for stride_slice now.";
  } else {
    steps = op->GetAttr<std::vector<int>>("strides");
  }
  // Decrease axis
  std::vector<int> decrease_axis;
  if (op->HasAttr("decrease_axis")) {
    decrease_axis = op->GetAttr<std::vector<int>>("decrease_axis");
  }
  // Output
  auto out_name = op->Output("Out").front();

  // Slice operation
  auto slice_operand = converter->AddSliceOperation(
      input_operand, axes, starts, ends, steps, out_name);
  // Use squeeze to process decrease_axis(attr)
  if (!decrease_axis.empty()) {
    // Squeeze operation
    converter->AddSqueezeOperation(slice_operand, decrease_axis, out_name);
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
