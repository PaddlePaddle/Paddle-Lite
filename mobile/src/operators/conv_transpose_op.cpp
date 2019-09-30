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

#ifdef CONV_TRANSPOSE_OP

#include "operators/conv_transpose_op.h"

namespace paddle_mobile {
namespace operators {}
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(conv2d_transpose, ops::ConvOpTranspose);
#endif

#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(conv2d_transpose, ops::ConvOpTranspose);
#endif

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(conv2d_transpose, ops::ConvOpTranspose);
#endif

#endif
