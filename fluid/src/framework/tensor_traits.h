
/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "traits.h"
#include "common/types.h"

namespace paddle_mobile {
    namespace framework {
        template<typename TargetType, DataType datatype, typename LayeOutType>
        class Tensor;

        template <typename TensorT>
        struct TensorTraits {
            typedef typename TensorT::target_category target_category;
            typedef typename TensorT::target_type target_type;
            typedef typename TensorT::layout_category layout_category;
            typedef typename TensorT::layout_type layout_type;
            using layout_dims = std::integral_constant<int, 0>;
        };

// NCHW_C4, the last dim is always 4
        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, NCHW_C4> >
        {
            typedef typename Tensor<TargetType, datatype, NCHW_C4>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, NCHW_C4>::target_type target_type;
            typedef typename DataTrait<datatype>::dtype Dtype;
            typedef D5 layout_category;
            typedef NCHW_C4 layout_type;
            using layout_dims = std::integral_constant<int, 5>;
            using num_idx = std::integral_constant<int, 0>;
            using channel_idx = std::integral_constant<int, 1>;
            using height_idx = std::integral_constant<int, 2>;
            using width_idx = std::integral_constant<int, 3>;
            using k_idx = std::integral_constant<int, 4>;
            static int num(const Shape& shape) {
                return shape[0];
            }
            static int channel(const Shape& shape) {
                return shape[1] * 4;
            }
            static int height(const Shape& shape) {
                return shape[2];
            }
            static int width(const Shape& shape) {
                return shape[3];
            }
            static int depth(const Shape& shape) {
                return shape[4];
            }
        };

        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, NHWC> >
        {
            typedef typename Tensor<TargetType, datatype, NHWC>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, NHWC>::target_type target_type;
            typedef D4 layout_category;
            typedef NHWC layout_type;
            using layout_dims = std::integral_constant<int, 4>;
            using num_idx = std::integral_constant<int, 0>;
            using channel_idx = std::integral_constant<int, 3>;
            using height_idx = std::integral_constant<int, 1>;
            using width_idx = std::integral_constant<int, 2>;
            static int num(const Shape& shape){
                return shape[0];
            }
            static int channel(const Shape& shape){
                return shape[3];
            }
            static int height(const Shape& shape){
                return shape[1];
            }
            static int width(const Shape& shape){
                return shape[2];
            }
        };

        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, NHW> >
        {
            typedef typename Tensor<TargetType, datatype, NHW>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, NHW>::target_type target_type;
            typedef D3 layout_category;
            typedef NHW layout_type;
            using layout_dims = std::integral_constant<int, 3>;
            using num_idx = std::integral_constant<int, 0>;
            using channel_idx = std::integral_constant<int, -1>;
            using height_idx = std::integral_constant<int, 1>;
            using width_idx = std::integral_constant<int, 2>;
            static int num(const Shape& shape){
                return shape[0];
            }
            static int channel(const Shape& shape){
                return 1;
            }
            static int height(const Shape& shape){
                return shape[1];
            }
            static int width(const Shape& shape){
                return shape[2];
            }
        };

        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, NW> >
        {
            typedef typename Tensor<TargetType, datatype, NW>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, NW>::target_type target_type;
            typedef D2 layout_category;
            typedef NW layout_type;
            using layout_dims = std::integral_constant<int, 2>;
            using num_idx = std::integral_constant<int, 0>;
            using channel_idx = std::integral_constant<int, -1>;
            using height_idx = std::integral_constant<int, -1>;
            using width_idx = std::integral_constant<int, 1>;
            static int num(const Shape& shape){
                return shape[0];
            }
            static int channel(const Shape& shape){
                return 1;
            }
            static int height(const Shape& shape){
                return 1;
            }
            static int width(const Shape& shape){
                return shape[2];
            }
        };

        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, HW> >
        {
            typedef typename Tensor<TargetType, datatype, HW>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, HW>::target_type target_type;
            typedef D2 layout_category;
            typedef HW layout_type;
            using layout_dims = std::integral_constant<int, 2>;
            using num_idx = std::integral_constant<int, -1>;
            using channel_idx = std::integral_constant<int, -1>;
            using height_idx = std::integral_constant<int, 0>;
            using width_idx = std::integral_constant<int, 1>;
            static int num(const Shape& shape){
                return 1;
            }
            static int channel(const Shape& shape){
                return 1;
            }
            static int height(const Shape& shape){
                return shape[0];
            }
            static int width(const Shape& shape){
                return shape[1];
            }
        };

        template<typename TargetType, DataType datatype>
        struct TensorTraits<Tensor<TargetType, datatype, W> >
        {
            typedef typename Tensor<TargetType, datatype, W>::target_category  target_category;
            typedef typename Tensor<TargetType, datatype, W>::target_type target_type;
            typedef D1 layout_category;
            typedef HW layout_type;
            using layout_dims = std::integral_constant<int, 1>;
            using num_idx = std::integral_constant<int, -1>;
            using channel_idx = std::integral_constant<int, -1>;
            using height_idx = std::integral_constant<int, -1>;
            using width_idx = std::integral_constant<int, 1>;
            static int num(const Shape& shape){
                return 1;
            }
            static int channel(const Shape& shape){
                return 1;
            }
            static int height(const Shape& shape){
                return 1;
            }
            static int width(const Shape& shape){
                return shape[0];
            }
        };





    }
}
