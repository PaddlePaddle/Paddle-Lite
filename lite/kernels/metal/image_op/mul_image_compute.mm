// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/metal/image_op/mul_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void MulImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.output->dims();
    auto input_dims = param.x->dims();

    auto s1 = 1, s2 = 1;
    for (int i = 0; i < param.x_num_col_dims; i++) {
        s1 *= input_dims[i];
    }

    for (int i = param.x_num_col_dims; i < input_dims.size(); i++) {
        s2 *= input_dims[i];
    }

    input_buffer_x_ = param.x->template data<P, MetalImage>();
    input_buffer_y_ = param.y->template data<P, MetalImage>();

    std::vector<int> nhwc = {0, 1, 2, 3};
    this->input_x_mul_dim_ = DDimLite({s1, s2});
    assert(input_buffer_y_->transpose_ == nhwc && input_buffer_y_->tensor_dim_.size() == 2 &&
           s2 == input_buffer_y_->tensor_dim_[0]);

    output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

    if (input_dims.size() != 2 || input_buffer_x_->transpose_ != nhwc) {
        insert_shape = true;
        std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
        reshape_ctx->template As<ContextMetal>().InitOnce();
        operators::ReshapeParam reshape_param;
        reshape_param.x = param.x;
        reshape_param.excepted_transpose_ = nhwc;
        shape_out_dev.Resize(this->input_x_mul_dim_.Vectorize());
        reshape_param.output = &shape_out_dev;
        reshape_.SetContext(std::move(reshape_ctx));
        reshape_.SetParam(reshape_param);
        reshape_.PrepareForRun();
    }

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "mul";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "mul_half";
    }
    assert(!function_name.empty());

    kernel_ = metal_context_->GetKernel(*device, function_name.c_str());
    queue_ = metal_context_->GetDefaultQueue(*device);
}

template <typename P, PrecisionType PTYPE>
void MulImageCompute<P, PTYPE>::Run() {
    const auto& param = this->template Param<param_t>();
    auto input_dims = param.x->dims();
    auto output_dims = param.output->dims();

    MetalUint output_width = output_buffer_->image().width;
    MetalUint output_height = output_buffer_->image().height;
    MetalUint output_array_length = output_buffer_->image().arrayLength;

    MetalUint3 global_work_size = {output_width, output_height, output_array_length};
    if (insert_shape) {
        reshape_.Run();
        auto encoder =
            std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
        auto shape_buffer = shape_out_dev.template data<P, MetalImage>();
        [encoder->metal_command_encoder_ setTexture:(shape_buffer->image()) atIndex:(0)];
        [encoder->metal_command_encoder_ setTexture:(input_buffer_y_->image()) atIndex:(1)];
        [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
        kernel_->Execute(*encoder, global_work_size, false);
    } else {
        auto encoder =
            std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
        [encoder->metal_command_encoder_ setTexture:(input_buffer_x_->image()) atIndex:(0)];
        [encoder->metal_command_encoder_ setTexture:(input_buffer_y_->image()) atIndex:(1)];
        [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
        kernel_->Execute(*encoder, global_work_size, false);
    }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::MulImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::MulImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::MulImageCompute<float, PRECISION(kFloat)> MetalMulFp32;
typedef paddle::lite::kernels::metal::MulImageCompute<MetalHalf, PRECISION(kFP16)> MetalMulFp16;

REGISTER_LITE_KERNEL(mul, kMetal, kFloat, kMetalTexture2DArray, MetalMulFp32, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(mul, kMetal, kFP16, kMetalTexture2DArray, MetalMulFp16, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();