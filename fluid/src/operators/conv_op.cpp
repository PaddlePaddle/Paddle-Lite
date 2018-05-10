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

#include "conv_op.h"
#include "framework/operator.h"
#include "framework/op_proto_maker.h"
#include "framework/shape_inference.h"
#include "framework/data_type.h"

namespace paddle_mobile {
namespace operators {

//    class ConvOp : public framework::OperatorWithKernel {
//    public:
//        void InferShape(framework::InferShapeContext* ctx) const override {};
//    protected:
//        framework::OpKernelType GetExpectedKernelType(
//                const framework::ExecutionContext& ctx) const override {};
//    };
//
//    class ConvOpMaker : public framework::OpProtoAndCheckerMaker {
//
//    };
template <typename Dtype, typename T>
void ConvOp<Dtype, T>::InferShape(framework::InferShapeContext* ctx) const {
//    PADDLE_ENFORCE(ctx->HasInput("Input"),
//                   "Input(Input) of ConvOp should not be null.");
//    PADDLE_ENFORCE(ctx->HasInput("Filter"),
//                   "Input(Filter) of ConvOp should not be null.");
//    PADDLE_ENFORCE(ctx->HasOutput("Output"),
//                   "Output(Output) of ConvOp should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    int groups = ctx->Attrs().Get<int>("groups");
    std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");

//    PADDLE_ENFORCE(in_dims.size() == 4 || in_dims.size() == 5,
//                   "Conv intput should be 4-D or 5-D tensor.");
//    PADDLE_ENFORCE_EQ(
//            in_dims.size(), filter_dims.size(),
//            "Conv input dimension and filter dimension should be the same.");
//    PADDLE_ENFORCE(
//            in_dims.size() - strides.size() == 2U,
//            "Conv input dimension and strides dimension should be consistent.");
//    PADDLE_ENFORCE_EQ(
//            paddings.size(), strides.size(),
//            "Conv paddings dimension and Conv strides dimension should be the same.");
//
//    PADDLE_ENFORCE_EQ(in_dims[1], filter_dims[1] * groups,
//                      "The number of input channels should be equal to filter "
//                              "channels * groups.");
//
//    PADDLE_ENFORCE_EQ(
//            filter_dims[0] % groups, 0,
//            "The number of output channels should be divided by groups.");

    std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
    for (size_t i = 0; i < strides.size(); ++i) {
        output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                              dilations[i], paddings[i],
                                              strides[i]));
    }
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");
}

template <typename DType, typename T>
framework::OpKernelType ConvOp<DType, T>::GetExpectedKernelType(
        const framework::ExecutionContext<DType>& ctx) const {
//#ifdef PADDLE_WITH_CUDA
//    if (platform::CanCUDNNBeUsed(ctx)) {
//library = framework::LibraryType::kCUDNN;
//}
//#endif
//#ifdef PADDLE_WITH_MKLDNN
//    if (library == framework::LibraryType::kPlain &&
//platform::CanMKLDNNBeUsed(ctx)) {
//library = framework::LibraryType::kMKLDNN;
//}
//#endif

    auto input_data_type =
            framework::ToDataType(ctx.template Input<Tensor>("Input")->type());
    auto filter_data_type =
            framework::ToDataType(ctx.template Input<Tensor>("Filter")->type());
//    PADDLE_ENFORCE_EQ(input_data_type, filter_data_type,
//                      "input and filter data type should be consistent");

//    if (input_data_type == framework::proto::VarType::FP16) {
//        PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
//                          "float16 can only be used when CUDNN is used");
//    }

    std::string data_format = ctx.template Attr<std::string>("data_format");
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::DataLayout layout = framework::StringToDataLayout(data_format);
    return framework::OpKernelType(input_data_type, layout);
}

} // operators
} // paddle_mobile
