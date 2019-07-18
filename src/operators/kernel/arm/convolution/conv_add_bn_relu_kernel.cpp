/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef FUSION_CONVADDBNRELU_OP

#include "operators/kernel/conv_add_bn_relu_kernel.h"
#include <cmath>
#include "framework/context.h"
#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include "operators/math/element_wise.h"
#include "operators/math/gemm/gemm1x1s1.h"
#include "operators/math/slidingwindow_utils.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNReluKernel<CPU, float>::Init(
    FusionConvAddBNReluParam<CPU> *param) {
  const Tensor *mean = param->InputMean();
  const Tensor *variance = param->InputVariance();
  const Tensor *scale = param->InputScale();
  const Tensor *bias = param->InputBias();
  const Tensor *bias1 = param->Bias();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();
  auto bias1_ptr = bias1->data<float>();

  const int C = mean->numel();
  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }

  Variable *scale_var = param->GetScope()->Var();
  Variable *bias_var = param->GetScope()->Var();
  LoDTensor *new_scale = scale_var->GetMutable<LoDTensor>();
  LoDTensor *new_bias = bias_var->GetMutable<LoDTensor>();
  float *new_scale_ptr = new_scale->mutable_data<float>({C});
  float *new_bias_ptr = new_bias->mutable_data<float>({C});
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] + (bias1_ptr[i] - mean_ptr[i]) *
                                        inv_std_ptr[i] * scale_ptr[i];
  }
  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  InitBaseConvKernel(param);

  // try to use faster depthwise conv
  switch (param->ExecMode()) {
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT:
      use_slidingwindow_add_bn_relu = true;
      break;
    case ConvParam<CPU>::EXEC_GEMM1x1s1_FLOAT:
      use_gemm_add_bn_relu = true;
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2_FLOAT:
      const std::vector<int> &paddings = param->Paddings();
      const std::vector<int> &strides = param->Strides();
      if (paddings.size() == 2 && paddings[0] == paddings[1] &&
          strides.size() == 2 && strides[0] == strides[1]) {
        int pad = paddings[0];
        int stride = strides[0];
        const int win = param->Input()->dims()[3];
        if (pad == 1) {
          if (stride == 1) {
            could_use_faster_depthwise_conv_ = true;
          } else if (stride == 2 && win > 7) {
            could_use_faster_depthwise_conv_ = true;
          }
        }
      }
      break;
  }

  if (could_use_faster_depthwise_conv_ || use_gemm_add_bn_relu ||
      use_slidingwindow_add_bn_relu) {
    auto filter_data = param->Filter()->data<float>();
    auto filter_dim = param->Filter()->dims();
    int len = 1;
    for (int i = 0; i < filter_dim.size(); i++) {
      len *= filter_dim[i];
    }
    int batch = filter_dim[0];
    int step = len / batch;
    for (int i = 0; i < batch; i++) {
      for (int k = 0; k < step; k++) {
        filter_data[i * step + k] =
            filter_data[i * step + k] * new_scale_ptr[i];
      }
    }
    if (use_gemm_add_bn_relu) {
      ARMArch arch = framework::CPUContext::Context()->get_arch();
      math::gemm1x1s1_transform_weight(*param->Filter(), *param->Output(),
                                       param->transformed_filter_,
                                       param->groups, arch);
    }
    if (use_slidingwindow_add_bn_relu) {
      math::slidingwindow_transform_weight<float>(*param->Filter(),
                                                  param->transformed_filter_);
    }
  }

  return true;
}

template <>
void ConvAddBNReluKernel<CPU, float>::Compute(
    const FusionConvAddBNReluParam<CPU> &param) {
  bool fusion_has_been_computed = false;
  switch (param.ExecMode()) {
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2_FLOAT:
      if (could_use_faster_depthwise_conv_) {
        FasterDepthwiseConv3x3_bias_relu(param, param.NewBias()->data<float>(),
                                         true);
        fusion_has_been_computed = true;
      } else {
        DepthwiseConv3x3<float, float>(param);
      }
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
    case ConvParam<CPU>::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_GEMM1x1s1_FLOAT:
      fusion_has_been_computed = true;
      GemmConv1x1s1<float, float>(param, param.NewBias()->data<float>(), true,
                                  true);
      break;
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT:
      SlidingwindowConv3x3<float, float>(param, param.NewBias()->data<float>(),
                                         true, true);
      fusion_has_been_computed = true;
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
  if (!fusion_has_been_computed) {
    math::ScaleAddChannelWise<RELU>(param.Output(), param.NewScale(),
                                    param.NewBias(), param.Output());
  }
}

template class ConvAddBNReluKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
