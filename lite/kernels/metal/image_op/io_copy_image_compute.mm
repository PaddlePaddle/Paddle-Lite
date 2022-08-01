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

#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/backends/metal/target_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include <sys/time.h>

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

/*
 * This kernel copies a tensor from host to Metal Texture(NHWC).
 */
class IoCopyHostToMetalTexture
    : public KernelLite<TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)> {
   public:
    void PrepareForRun() override {
        auto& context = ctx_->As<MTLContext>();
        metal_context_ = (MetalContext*)context.context();

        auto& param = Param<operators::IoCopyParam>();
        CHECK(param.x->target() == TARGET(kHost) || param.x->target() == TARGET(kARM) ||
              param.x->target() == TARGET(kX86));

        auto input_dims = param.x->dims();
        auto src = param.x->template data<float>();
        // scene: have not metal kernel, so use CPU kernel then return to GPU
        if (input_dims.size() == 4 || input_dims.size() == 3) {
            output_buffer_ = param.y->template mutable_data<MetalHalf, MetalImage>(
                metal_context_, param.y->dims());
        }
        // scene: op params, resident memory can be initialized once
        else {
            output_buffer_ = param.y->template mutable_data<MetalHalf, MetalImage>(
                metal_context_, param.y->dims(), {0, 1, 2, 3}, false);
            output_buffer_->src_tensor_ = (void*)param.x;
            output_buffer_->CopyFromNCHW<float>(src);
            function_name_ = "host_to_metal-prepare";
        }
    }

    void Run() override {
        auto& param = Param<operators::IoCopyParam>();
        auto input_dims = param.x->dims();
        auto output_dims = param.y->dims();

        if (input_dims.size() == 4 || input_dims.size() == 3) {
            auto src = param.x->template data<float>();
            output_buffer_->CopyFromNCHW<float>(src);
        } else {
        }
    }

    void SaveOutput() {
        if (function_name_ == "buf_to_tex_c_n") {
            MetalDebug::SaveOutput(function_name_, output_buffer_);
        }
    };

    virtual ~IoCopyHostToMetalTexture() {
        TargetWrapperMetal::FreeImage(output_buffer_);
    }

    std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
        std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
        *res = [](const std::map<std::string, const Type*>& inputs,
            const std::string& out) -> const Type* {
            CHECK(!inputs.empty());
            auto* type = inputs.at("Input");
            CHECK(type->target() == TARGET(kHost));

            auto out_place = type->place();
            out_place.target = TARGET(kMetal);
            auto* out_type = Type::Get(type->id(),
                out_place.target,
                out_place.precision,
                out_place.layout,
                out_place.device);
            return out_type;
        };
        return res;
    }

    std::string doc() const override {
        return "Copy IO from HOST to Metal";
    }

    MetalImage* output_buffer_{nullptr};
    std::shared_ptr<MetalBuffer> src_buffer_;

    id<MTLComputePipelineState> pipline_;
    std::string function_name_;
    MetalContext* metal_context_;
};

/*
 * This kernel copies a tensor from Metal to host space.
 */
class IoCopykMetalTextureToHost
    : public KernelLite<TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)> {
   public:
    void PrepareForRun() override {
        auto& context = ctx_->As<MTLContext>();
        metal_context_ = (MetalContext*)context.context();
    }

    void SaveOutput() {
        auto& param = this->Param<operators::IoCopyParam>();
        auto src = param.y->template data<float>();
        MetalDebug::print_float("metal_to_host", (float*)src, (int)param.y->dims().production());
    };

    void Run() override {
        auto backend = (__bridge MetalContextImp*)metal_context_->backend();
        [backend waitAllCompleted];

        auto& param = this->Param<operators::IoCopyParam>();
        CHECK(param.x->target() == TARGET(kMetal));
        auto src = param.x->template data<MetalHalf, MetalImage>();

        auto mem_size = param.x->dims().production() * sizeof(float);
        auto data = param.y->template mutable_data<float>(TARGET(kHost), mem_size);
        src->template CopyToNCHW<float>(data);
    }

    std::string doc() const override {
        return "Copy IO from kMetal to HOST";
    }

    MetalContext* metal_context_;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::IoCopyHostToMetalTexture,
    host_to_device_image)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::IoCopykMetalTextureToHost,
    device_image_to_host)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::IoCopyHostToMetalTexture,
    host_to_device_image)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::IoCopykMetalTextureToHost,
    device_image_to_host)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .Finalize();
