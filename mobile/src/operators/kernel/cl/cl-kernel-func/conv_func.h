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

#if defined(CONV_OP) || defined(CONV_TRANSPOSE_OP)

#pragma once

#include "framework/cl/cl_helper.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

inline int maptofactor(int i, int factor) { return (i + factor - 1) / factor; }

template <int tile, int kernel>
void winograd_transform_weight(framework::CLHelper *cl_helper,
                               framework::CLImage *weight);

template <int tile, int kernel>
void WinogradConv3x3(framework::CLHelper *cl_helper,
                     const ConvParam<GPU_CL> &param, bool ifRelu = false,
                     const framework::CLImage *biase = nullptr,
                     const framework::CLImage *new_scale = nullptr,
                     const framework::CLImage *new_bias = nullptr);

void ConvAddBnRelu(framework::CLHelper *cl_helper,
                   const ConvParam<GPU_CL> &param, bool ifRelu = false,
                   const framework::CLImage *biase = nullptr,
                   const framework::CLImage *new_scale = nullptr,
                   const framework::CLImage *new_bias = nullptr);

void DWConvAddBnRelu(framework::CLHelper *cl_helper,
                     const ConvParam<GPU_CL> &param, bool ifRelu = false,
                     const framework::CLImage *biase = nullptr,
                     const framework::CLImage *new_scale = nullptr,
                     const framework::CLImage *new_bias = nullptr);

void SWConvAddBnRelu(framework::CLHelper *cl_helper,
                     const ConvParam<GPU_CL> &param, bool ifRelu = false,
                     const framework::CLImage *biase = nullptr,
                     const framework::CLImage *new_scale = nullptr,
                     const framework::CLImage *new_bias = nullptr);
void DWConvTransposeAddBnRelu(framework::CLHelper *cl_helper,
                              const ConvTransposeParam<GPU_CL> &param,
                              bool ifRelu = false,
                              const framework::CLImage *biase = nullptr,
                              const framework::CLImage *new_scale = nullptr,
                              const framework::CLImage *new_bias = nullptr);
void ConvTransposeAddBnRelu(framework::CLHelper *cl_helper,
                            const ConvTransposeParam<GPU_CL> &param,
                            bool ifRelu = false,
                            const framework::CLImage *biase = nullptr,
                            const framework::CLImage *new_scale = nullptr,
                            const framework::CLImage *new_bias = nullptr);

}  // namespace operators
}  // namespace paddle_mobile

#endif
