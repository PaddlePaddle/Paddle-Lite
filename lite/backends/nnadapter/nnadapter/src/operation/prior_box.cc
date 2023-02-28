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

#include "operation/prior_box.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidatePriorBox(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PreparePriorBox(core::Operation* operation) {
  PRIOR_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto& output_type0 = output_operands[0]->type;
  auto& output_type1 = output_operands[1]->type;
  CopyOperandTypeWithQuantParams(&output_type0, input_type);
  CopyOperandTypeWithQuantParams(&output_type1, input_type);
  auto ExpandAspectRatios = [&](std::vector<float>& aspect_ratios_,
                                bool flip,
                                std::vector<float>* output_aspect_ratior) {
    constexpr float epsilon = 1e-6;
    output_aspect_ratior->clear();
    output_aspect_ratior->push_back(1.0f);
    for (size_t i = 0; i < aspect_ratios_.size(); ++i) {
      float ar = aspect_ratios_[i];
      bool already_exist = false;
      for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
        if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
          already_exist = true;
          break;
        }
      }
      if (!already_exist) {
        output_aspect_ratior->push_back(ar);
        if (flip) {
          output_aspect_ratior->push_back(1.0f / ar);
        }
      }
    }
  };
  auto infer_output_shape = [&](const int32_t* input_dimensions,
                                int32_t* output_dimensions0,
                                int32_t* output_dimensions1) {
    std::vector<float> aspect_ratios_vec;
    float* aspect_ratios_data =
        reinterpret_cast<float*>(aspect_ratios_operand->buffer);
    uint32_t aspect_ratios_size = aspect_ratios_operand->length / sizeof(float);
    std::vector<float> aspect_ratios_(aspect_ratios_data,
                                      aspect_ratios_data + aspect_ratios_size);
    ExpandAspectRatios(aspect_ratios_,
                       *reinterpret_cast<bool*>(flip_operand->buffer),
                       &aspect_ratios_vec);
    size_t num_priors =
        aspect_ratios_vec.size() * min_sizes_operand->length / sizeof(float) +
        max_sizes.size();
    std::vector<int64_t> dim_vec(4);
    output_dimensions0[0] = input_dimensions[2];
    output_dimensions0[1] = input_dimensions[3];
    output_dimensions0[2] = num_priors;
    output_dimensions0[3] = 4;
    output_dimensions1[0] = input_dimensions[2];
    output_dimensions1[1] = input_dimensions[3];
    output_dimensions1[2] = num_priors;
    output_dimensions1[3] = 4;
  };
  infer_output_shape(input_type.dimensions.data,
                     output_type0.dimensions.data,
                     output_type1.dimensions.data);
  NNADAPTER_VLOG(5) << "boxes: " << OperandToString(boxes_operand);
  NNADAPTER_VLOG(5) << "Variances: " << OperandToString(Variances_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecutePriorBox(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
