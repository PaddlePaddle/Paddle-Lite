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
// limitations under the License.i

#include "lite/core/op_registry.h"
#include "lite/operators/activation_ops.h"

// Extra activation ops
REGISTER_LITE_OP(square, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu_clipped, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(swish, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(log, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(exp, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(abs, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(floor, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(hard_sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(sqrt, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(rsqrt, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(softsign, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(gelu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(hard_swish, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(reciprocal, paddle::lite::operators::ActivationOp);
