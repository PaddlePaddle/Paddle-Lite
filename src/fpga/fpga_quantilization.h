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
#pragma once

#include "common/types.h"
#include "framework/lod_tensor.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace fpga {

template <typename Dtype>
static void chw_to_hwc(Dtype* data_in, Dtype* data_out, int num, int channel,
                       int height, int width);

// template <typename Dtype>
void quantify_filter(framework::Tensor* filter);

}  // namespace fpga
}  // namespace paddle_mobile
