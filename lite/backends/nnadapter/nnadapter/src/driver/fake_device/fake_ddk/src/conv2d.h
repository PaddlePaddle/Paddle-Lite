// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "fake_ddk/operator.h"
#include "fake_ddk/tensor.h"
#include "utility.h"  // NOLINT

namespace fake_ddk {

int conv2d(Tensor* input_tensor,
           Tensor* filter_tensor,
           Tensor* bias_tensor,
           Tensor* output_tensor,
           Conv2DAttr* conv2d_attr);

}  // namespace fake_ddk
