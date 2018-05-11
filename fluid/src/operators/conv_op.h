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

#include "framework/operator.h"
#include "math/vol2col.h"
#include "math/im2col.h"
#include "math/math_function.h"

namespace paddle_mobile {
namespace operators {

    using Tensor = framework::Tensor;

    // Base convolution operator definations for other conv
    // like operators to reuse the implementation.
    inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                              int padding, int stride) {
        const int dkernel = dilation * (filter_size - 1) + 1;
        int output_size = (input_size + 2 * padding - dkernel) / stride + 1;

        return output_size;
    }
    inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations) {
        bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
        for (size_t j = 0; j < strides.size(); ++j) {
            filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
            strides_1 = strides_1 && (strides[j] == 1);
            padding_0 = padding_0 && (paddings[j] == 0);
            dilation_1 = dilation_1 && (dilations[j] == 1);
        }
        return !(filter_1 && strides_1 && padding_0 && dilation_1);
    }

    template <typename DeviceType, typename T>
    class ConvOp : public framework::OperatorWithKernel<DeviceType> {
    public:
        using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;
        void InferShape(framework::InferShapeContext* ctx) const override;

    protected:
        framework::OpKernelType GetExpectedKernelType(
                const framework::ExecutionContext<DeviceType>& ctx) const override;
    };

    template <typename DeviceType, typename T>
    class GemmConvKernel : public framework::OpKernel<DeviceType> {
    public:
        void Compute(const framework::ExecutionContext<DeviceType>& context) const override {
            const Tensor* input = context.template Input<Tensor>("Input");
          if (!input){
            std::cout << "is null" << std::endl;
          }

            // The filter will be reshaped in the calculations,
            // so here use an assignment operation,
            // that avoids modifying the variable in the Scope.
            Tensor filter = *context.template Input<Tensor>("Filter");
            Tensor* output = context.template Output<Tensor>("Output");
//            output->mutable_data<T>(context.GetPlace());

            int groups = context.template Attr<int>("groups");

            std::vector<int> strides = context.template Attr<std::vector<int>>("strides");
            std::vector<int> paddings = context.template Attr<std::vector<int>>("paddings");
            std::vector<int> dilations = context.template Attr<std::vector<int>>("dilations");

          std::cout << " compute end get Attrs " << strides[0] << std::endl;

            const int batch_size = static_cast<int>(input->dims()[0]);

            // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
            std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
            // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
            std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

            // use col_shape in the im2col calculation
            // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
            // o_h, o_w}
            size_t data_dim = filter_shape_vec.size() - 2;
            std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
            col_shape_vec[0] = input->dims()[1] / groups;
            for (size_t j = 0; j < data_dim; ++j) {
                col_shape_vec[j + 1] = filter_shape_vec[j + 2];
                col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
            }
            framework::DDim col_shape(framework::make_ddim(col_shape_vec));

            // use col_matrix_shape in the gemm calculation
            // size: (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d *
            // o_h * o_w)
            framework::DDim col_matrix_shape =
                    framework::flatten_to_2d(col_shape, data_dim + 1);

            bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
            Tensor col;
            // col_matrix shares the same piece of data with col,
            // but will be reshaped into a two-dimensional matrix shape
            // to call the matrix multiplication interface.
            Tensor col_matrix;
            if (is_expand) {
                col.mutable_data<T>(col_shape);
                col_matrix.ShareDataWith(col);
                col_matrix.Resize(col_matrix_shape);
            }

            framework::DDim input_shape = framework::slice_ddim(
                    input->dims(), 1, static_cast<int>(input->dims().size()));

            framework::DDim filter_matrix_shape = {filter.dims()[0],
                                                   filter.numel() / filter.dims()[0]};
            filter.Resize(filter_matrix_shape);

          std::cout << " input dim " << input->dims() << std::endl;

          std::cout << " output dim " << output->dims() << std::endl;

            framework::DDim output_matrix_shape = {
                    output->dims()[1],
                    output->numel() / (output->dims()[0] * output->dims()[1])};

            // convolution operator: im2col(or vol2col) + gemm
            int in_step = static_cast<int>(input->dims()[1]) / groups;
            int out_step = static_cast<int>(output->dims()[1]) / groups;

            math::Vol2ColFunctor<DeviceType, T> vol2col;
            math::Im2ColFunctor<math::ColFormat::kCFO, DeviceType, T> im2col;

//            auto& dev_ctx = context.template device_context<DeviceContext>();
            for (int i = 0; i < batch_size; i++) {
                Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
                Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

                for (int g = 0; g < groups; g++) {
                    Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

                    if (!is_expand) {
                        col.ShareDataWith(in_slice);
                        col_matrix.ShareDataWith(col);
                        col_matrix.Resize(col_matrix_shape);
                    } else if (data_dim == 2U) {
                        // im2col
                        im2col(in_slice, dilations, strides,
                               std::vector<int>{paddings[0], paddings[1], paddings[0],
                                                paddings[1]},
                               &col);
                    } else if (data_dim == 3U) {
                        // vol2col
                        vol2col(in_slice, dilations, strides, paddings, &col);
                    }

                    // gemm
                    Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
                    Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
                    math::matmul<T>(filter_slice, false, col_matrix,
                                                   false, T(1.0), &out_slice, T(0.0));
                }
            }
        }
    };
} // operators
} // paddle_mobile
