// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "compute_param.h"  // NOLINT
#include "lite/operators/op_params.h"
#include "log_lite.h"  // NOLINT

namespace paddle {
namespace lite_api {
void *ActivationParam::AttachRawParam() {
  //! necessary check
  LCHECK(X, "ActivationParam must set input tensor: X\n");
  LCHECK(Out, "ActivationParam must set output tensor: Out\n");

  auto *raw_act_param = new lite::operators::ActivationParam();
  // Tensor
  raw_act_param->X = static_cast<const lite::Tensor *>(X->GetRawTensor());
  raw_act_param->Out = static_cast<lite::Tensor *>(Out->GetRawTensor());
  raw_act_param->Prelu_alpha =
      Prelu_alpha ? static_cast<lite::Tensor *>(Prelu_alpha->GetRawTensor())
                  : nullptr;

  raw_act_param->active_type = active_type;
  raw_act_param->has_active = has_active;
  raw_act_param->Leaky_relu_alpha = Leaky_relu_alpha;
  raw_act_param->Relu_clipped_coef = Relu_clipped_coef;
  raw_act_param->Prelu_mode = Prelu_mode;
  raw_act_param->Swish_beta = Swish_beta;
  raw_act_param->hard_sigmoid_slope = hard_sigmoid_slope;
  raw_act_param->hard_sigmoid_offset = hard_sigmoid_offset;
  raw_act_param->hard_swish_scale = hard_swish_scale;
  raw_act_param->hard_swish_offset = hard_swish_offset;
  raw_act_param->hard_swish_threshold = hard_swish_threshold;

  return raw_act_param;
}

void *ConvParam::AttachRawParam() {
  //! necessary check
  LCHECK(x, "ConvParam must set input tensor: x\n");
  LCHECK(filter, "ConvParam must set filter tensor: filter\n");
  LCHECK(output, "ConvParam must set output tensor: output\n");
  if (enable_int8 && out_ptype == PRECISION(kFloat)) {
    LCHECK_NE(input_scale, 0.f, "int8 conv out float, must has input scale\n");
    LCHECK(!weight_scale.empty(),
           "int8 conv out float, must has weights scale\n");
  } else if (enable_int8 && out_ptype == PRECISION(kInt8)) {
    LCHECK_NE(input_scale, 0.f, "int8 conv out int8, must has input scale\n");
    LCHECK_NE(output_scale, 0.f, "int8 conv out int8, must has output scale\n");
    LCHECK(!weight_scale.empty(),
           "int8 conv out int8, must has weights scale\n");
  }

  auto *raw_conv_param = new lite::operators::ConvParam();
  // Tensor
  raw_conv_param->x = static_cast<lite::Tensor *>(x->GetRawTensor());
  raw_conv_param->filter = static_cast<lite::Tensor *>(filter->GetRawTensor());
  raw_conv_param->output = static_cast<lite::Tensor *>(output->GetRawTensor());
  raw_conv_param->bias =
      bias ? static_cast<lite::Tensor *>(bias->GetRawTensor()) : nullptr;
  raw_conv_param->residualData =
      residualData ? static_cast<lite::Tensor *>(residualData->GetRawTensor())
                   : nullptr;

  // activation param
  raw_conv_param->activation_param.active_type = activation_param.active_type;
  raw_conv_param->activation_param.has_active = activation_param.has_active;
  raw_conv_param->activation_param.Relu_clipped_coef =
      activation_param.Relu_clipped_coef;
  raw_conv_param->activation_param.Leaky_relu_alpha =
      activation_param.Leaky_relu_alpha;
  raw_conv_param->activation_param.Swish_beta = activation_param.Swish_beta;
  raw_conv_param->activation_param.hard_sigmoid_slope =
      activation_param.hard_sigmoid_slope;
  raw_conv_param->activation_param.hard_sigmoid_offset =
      activation_param.hard_sigmoid_offset;
  raw_conv_param->activation_param.hard_swish_scale =
      activation_param.hard_swish_scale;
  raw_conv_param->activation_param.hard_swish_offset =
      activation_param.hard_swish_offset;
  raw_conv_param->activation_param.hard_swish_threshold =
      activation_param.hard_swish_threshold;

  // for int8
  raw_conv_param->enable_int8 = enable_int8;
  raw_conv_param->input_scale = input_scale;
  raw_conv_param->weight_scale = weight_scale;
  raw_conv_param->output_scale = output_scale;
  raw_conv_param->bit_length = bit_length;

  raw_conv_param->strides = strides;
  raw_conv_param->paddings = paddings;
  raw_conv_param->groups = groups;
  raw_conv_param->dilations = dilations;
  raw_conv_param->fuse_residual_connection = fuse_residual_connection;
  raw_conv_param->data_format = data_format;
  raw_conv_param->output_size = output_size;

  return raw_conv_param;
}

int ConvParam::GetKernelIndex() {
  if (enable_int8) {
    if (out_ptype == PRECISION(kFloat)) {
      return 1;
    } else if (out_ptype == PRECISION(kInt8)) {
      return 0;
    } else {
      LOGF("conv only support float and int8 precision\n");
    }
  } else {
    return 0;
  }
}
}  // namespace lite_api
}  // namespace paddle
