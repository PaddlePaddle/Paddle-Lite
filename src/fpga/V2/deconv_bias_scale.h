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

namespace paddle_mobile {
namespace fpga {
namespace deconv_bias_scale {

void deconv_bias_scale_expand(float** bias_scale_array, int num,
                              int sub_conv_n);

}  // namespace deconv_bias_scale
}  // namespace fpga
}  // namespace paddle_mobile
