/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "operators/kernel/elementwise_add_kernel.h"

namespace paddle_mobile {
    namespace operators {

        template <typename T> struct AddFunctor {
            inline T operator()(T a, T b) const { return a + b; }
        };

        template <>
        void ElementwiseAddKernel<CPU, float, ElementwiseAddParam>::Compute(
            const ElementwiseAddParam &param) const {
            const Tensor *input_x = param.InputX();
            const Tensor *input_y = param.InputY();
            Tensor *Out = param.Out();
            Out->mutable_data<float>();
            const int axis = param.Axis();
            ElementwiseComputeEx<AddFunctor<float>, float>(
                input_x, input_y, axis, AddFunctor<float>(), Out);
        }

        template class ElementwiseAddKernel<CPU, float, ElementwiseAddParam>;

    } // namespace operators
} // namespace paddle
