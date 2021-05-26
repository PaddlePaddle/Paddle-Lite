// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/metal/image_op/split_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void SplitImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto outputs = param.output;

    size_t num = outputs.size();

    input_buffer_ = param.x->template data<P, MetalImage>();
    for (int i = 0; i < num; i++) {
        auto output_dim = outputs[i]->dims();
        auto output_image = outputs[i]->template mutable_data<P, MetalImage>(output_dim);
        output_buffers_.emplace_back(output_image);
    }

    int axis = param.axis;
    auto* axis_tensor = param.axis_tensor;
    if (axis_tensor != nullptr) {
        auto* axis_tensor_data = axis_tensor->template data<int>();
        axis = axis_tensor_data[0];
    }
    if (axis < 0) {
        axis += input_buffer_->tensor_dim_.size();
    }
    auto rank = input_buffer_->tensor_dim_.size();
    auto axis_param =
        (int)(param.axis + input_buffer_->tensor_dim_.size() - input_buffer_->tensor_dim_.size());

    SplitMetalParam smp = {{
        (int)param.x->dims()[0],
        (int)param.x->dims()[1],
        (int)param.x->dims()[2],
        (int)param.x->dims()[3],
    }};
    smp.trans[0] = (int)(input_buffer_->transpose_[0]);
    smp.trans[1] = (int)(input_buffer_->transpose_[1]);
    smp.trans[2] = (int)(input_buffer_->transpose_[2]);
    smp.trans[3] = (int)(input_buffer_->transpose_[3]);

    for (int i = 0; i < 4; i++) {
        if (input_buffer_->transpose_[i] == smp.axis) {
            smp.axis = i;
            break;
        }
    }

    int vdim[4]{0, 0, 0, 0};
    for (int i = 0; i < num; i++) {
        vdim[i] = int(param.output[i]->dims()[param.axis]);
    }

    smp.vdim[0] = vdim[0];
    smp.vdim[1] = vdim[1];
    smp.vdim[2] = vdim[2];
    smp.vdim[3] = vdim[3];

    v_ = "normal";
    if (rank == 4) {
        if (smp.axis == 1) {
            v_ = "y";
        } else if (smp.axis == 2) {
            v_ = "x";
        } else if (smp.axis == 3 && input_buffer_->tensor_dim_[0] == 1) {
            auto vz = true;
            for (int i = 0; i < num; i++) {
                if (vdim[i] % 4 != 0) {
                    vz = false;
                    break;
                }
            }
            if (vz) {
                v_ = "z";
                smp.vdim[0] = vdim[0] / 4;
                smp.vdim[1] = vdim[1] / 4;
                smp.vdim[2] = vdim[2] / 4;
                smp.vdim[3] = vdim[3] / 4;
            } else {
                v_ = "zz";
            }
        }
    } else if (rank == 3) {
        if (smp.axis == 2) {
            v_ = "y";
        } else if (smp.axis == 3) {
            v_ = "x";
        }
    } else if (rank == 2) {
        if (smp.axis == 2) {
            v_ = "y";
        }
    }
    if (v_ == "normal") {
        throw std::logic_error("ERROR: unsupported split type");
    }

    param_buffer_ =
        metal_context_->CreateBuffer(*device, &smp, sizeof(smp), METAL_ACCESS_FLAG::CPUWriteOnly);

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name =
            "split_" + std::to_string(rank) + "_" + std::to_string(num) + "_" + v_ + "_float";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name =
            "split_" + std::to_string(rank) + "_" + std::to_string(num) + "_" + v_ + "_half";
    }
    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void SplitImageCompute<P, PTYPE>::Run() {
    auto output_width = input_buffer_->texture_width_;
    auto output_height = input_buffer_->texture_height_;
    auto output_array_length = input_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    NSUInteger image_index = 0;
    [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(image_index++)];
    for (auto item : output_buffers_)
        [encoder->metal_command_encoder_ setTexture:(item->image()) atIndex:(image_index++)];
    [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0)atIndex:(0)];

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::SplitImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::SplitImageCompute<MetalHalf, PRECISION(kFP16)>;
typedef paddle::lite::kernels::metal::SplitImageCompute<float, PRECISION(kFloat)> MetalSplitFp32;
typedef paddle::lite::kernels::metal::SplitImageCompute<MetalHalf, PRECISION(kFP16)> MetalSplitFp16;

REGISTER_LITE_KERNEL(split, kMetal, kFloat, kMetalTexture2DArray, MetalSplitFp32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(split, kMetal, kFP16, kMetalTexture2DArray, MetalSplitFp16, def)
    .BindInput(
        "X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})

    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput(
        "Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();