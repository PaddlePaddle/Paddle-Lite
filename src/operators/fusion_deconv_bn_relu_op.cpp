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

#ifdef FUSION_DECONVBNRELU_OP

#include "operators/fusion_deconv_bn_relu_op.h"

namespace paddle_mobile {
namespace operators {}
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
REGISTER_FUSION_MATCHER(fusion_deconv_bn_relu, ops::FusionDeconvBNReluMatcher);
#ifdef PADDLE_MOBILE_CPU
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(fusion_deconv_bn_relu, ops::FusionDeconvBNReluOp);
#endif

#endif
