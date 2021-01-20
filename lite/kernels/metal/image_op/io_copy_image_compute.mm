// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <sys/time.h>
#include "lite/backends/metal/target_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

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
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) || param.x->target() == TARGET(kARM) ||
          param.x->target() == TARGET(kX86));
    output_buffer_ = param.y->template mutable_data<float, metal_image>(param.y->dims());
    auto input_dims = param.x->dims();

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      std::string function_name = "";
      if (std::is_same<float, float>::value) {
        function_name = "buffer_to_texture_array_n_channel_kernel";
      } else if (std::is_same<float, metal_half>::value) {
        function_name = "buffer_to_texture_array_n_channel_kernel_half";
      }

      kernel_ = mtl_ctx->get_kernel(*device, function_name);
    }
  }

  void Run() override {
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    auto src = param.x->template data<float>();
    auto input_dims = param.x->dims();
    auto output_dims = param.y->dims();
    auto mem_size = param.x->dims().production() * sizeof(float);
    auto src_buffer_ = mtl_ctx->create_buffer(
        *device, const_cast<float*>(src), mem_size, METAL_ACCESS_FLAG::CPUWriteOnly);

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      auto output_width = output_buffer_->textureWidth_;
      auto output_height = output_buffer_->textureHeight_;
      auto output_array_length = output_buffer_->arrayLength_;

      auto& context = ctx_->As<MetalContext>();
      auto mtl_ctx = (metal_context*)context.context();
      auto mtl_dev = mtl_ctx->get_default_device();

      {
        auto queue = mtl_ctx->get_default_queue(*mtl_dev);
        metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                        static_cast<metal_uint>(output_height),
                                        static_cast<metal_uint>(output_array_length)};

        auto args = {metal_kernel_arg{src_buffer_}, metal_kernel_arg{output_buffer_}};
        kernel_->execute(*queue, global_work_size, 0, args);
        queue->wait_until_complete();
      }
    } else {
      output_buffer_->from_nchw<float>(src);
    }
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
      auto* out_type = Type::Get(
          type->id(), out_place.target, out_place.precision, out_place.layout, out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to Metal"; }

  metal_image* output_buffer_ = nullptr;
  std::shared_ptr<metal_kernel> kernel_;
};

/*
 * This kernel copies a tensor from Metal to host space.
 */
class IoCopykMetalTextureToHost
    : public KernelLite<TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)> {
 public:
  void Run() override {
    auto& param = this->Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kMetal));
    auto src = param.x->template data<float, metal_image>();

    auto mem_size = param.x->dims().production() * sizeof(float);
    auto data = param.y->template mutable_data<float>(TARGET(kHost), mem_size);
    src->template to_nchw<float>(data);
  }

  std::string doc() const override { return "Copy IO from kMetal to HOST"; }

  float d2h_duration_{0};
};

class IoCopyHostToMetalTextureHalf
    : public KernelLite<TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)> {
 public:
  void PrepareForRun() override {
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) || param.x->target() == TARGET(kARM) ||
          param.x->target() == TARGET(kX86));
    output_buffer_ = param.y->template mutable_data<metal_half, metal_image>(param.y->dims());
    auto input_dims = param.x->dims();

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      std::string function_name = "buffer_to_texture_array_n_channel_kernel_half";
      kernel_ = mtl_ctx->get_kernel(*device, function_name);
    }
  }

  void Run() override {
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    auto src = param.x->template data<float>();
    auto input_dims = param.x->dims();
    auto output_dims = param.y->dims();
    auto mem_size = param.x->dims().production() * sizeof(float);

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      auto src_buffer_ = mtl_ctx->create_buffer(
          *device, const_cast<float*>(src), mem_size, METAL_ACCESS_FLAG::CPUWriteOnly);
      auto output_width = output_buffer_->textureWidth_;
      auto output_height = output_buffer_->textureHeight_;
      auto output_array_length = output_buffer_->arrayLength_;

      auto& context = ctx_->As<MetalContext>();
      auto mtl_ctx = (metal_context*)context.context();
      auto mtl_dev = mtl_ctx->get_default_device();

      {
        auto queue = mtl_ctx->get_default_queue(*mtl_dev);
        metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                        static_cast<metal_uint>(output_height),
                                        static_cast<metal_uint>(output_array_length)};

        auto args = {metal_kernel_arg{src_buffer_}, metal_kernel_arg{output_buffer_}};
        kernel_->execute(*queue, global_work_size, 0, args);
        queue->wait_until_complete();
      }
    } else {
      output_buffer_->from_nchw<float>(src);
    }
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
      auto* out_type = Type::Get(
          type->id(), out_place.target, out_place.precision, out_place.layout, out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to Metal"; }

  metal_image* output_buffer_ = nullptr;
  std::shared_ptr<metal_kernel> kernel_;
};

/*
 * This kernel copies a tensor from Metal to host space.
 */
class IoCopykMetalTextureToHostHalf
    : public KernelLite<TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)> {
 public:
  void Run() override {
    auto& param = this->Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kMetal));
    auto src = param.x->template data<float, metal_image>();

    auto mem_size = param.x->dims().production() * sizeof(float);
    auto data = param.y->template mutable_data<float>(TARGET(kHost), mem_size);
    src->template to_nchw<float>(data);
  }

  std::string doc() const override { return "Copy IO from kMetal to HOST"; }

  float d2h_duration_{0};
};

class IoCopyHostToMetalTextureHalf2Half
    : public KernelLite<TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kMetalTexture2DArray)> {
 public:
  void PrepareForRun() override {
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) || param.x->target() == TARGET(kARM) ||
          param.x->target() == TARGET(kX86));
    output_buffer_ = param.y->template mutable_data<metal_half, metal_image>(param.y->dims());
    auto input_dims = param.x->dims();

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      std::string function_name = "buffer_to_texture_array_n_channel_kernel_half";
      kernel_ = mtl_ctx->get_kernel(*device, function_name);
    }
  }

  void Run() override {
    auto& context = ctx_->As<MetalContext>();
    auto mtl_ctx = (metal_context*)context.context();
    auto device = mtl_ctx->get_default_device();
    auto& param = Param<operators::IoCopyParam>();
    auto src = param.x->template data<metal_half>();
    auto input_dims = param.x->dims();
    auto output_dims = param.y->dims();
    auto mem_size = param.x->dims().production() * sizeof(metal_half);
    auto src_buffer_ = mtl_ctx->create_buffer(
        *device, const_cast<metal_half*>(src), mem_size, METAL_ACCESS_FLAG::CPUReadWrite);

    if ((input_dims.size() == 4 && input_dims[1] <= 4) ||
        (input_dims.size() == 3 && input_dims[0] <= 4)) {
      auto output_width = output_buffer_->textureWidth_;
      auto output_height = output_buffer_->textureHeight_;
      auto output_array_length = output_buffer_->arrayLength_;

      auto& context = ctx_->As<MetalContext>();
      auto mtl_ctx = (metal_context*)context.context();
      auto mtl_dev = mtl_ctx->get_default_device();

      {
        auto queue = mtl_ctx->get_default_queue(*mtl_dev);
        metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                        static_cast<metal_uint>(output_height),
                                        static_cast<metal_uint>(output_array_length)};

        auto args = {metal_kernel_arg{src_buffer_}, metal_kernel_arg{output_buffer_}};
        kernel_->execute(*queue, global_work_size, 0, args);
        queue->wait_until_complete();
      }
    } else {
      output_buffer_->from_nchw<metal_half>(src);
    }
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
      auto* out_type = Type::Get(
          type->id(), out_place.target, out_place.precision, out_place.layout, out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to Metal"; }

  metal_image* output_buffer_ = nullptr;
  std::shared_ptr<metal_kernel> kernel_;
};

/*
 * This kernel copies a tensor from Metal to host space.
 */
class IoCopykMetalTextureToHostHalf2Half
    : public KernelLite<TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)> {
 public:
  void Run() override {
    auto& param = this->Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kMetal));
    auto src = param.x->template data<metal_half, metal_image>();

    auto mem_size = param.x->dims().production() * sizeof(metal_half);
    auto data = param.y->template mutable_data<metal_half>(TARGET(kHost), mem_size);
    src->template to_nchw<metal_half>(data);
  }

  std::string doc() const override { return "Copy IO from kMetal to HOST"; }

  float d2h_duration_{0};
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
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kNCHW))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopykMetalTextureToHost,
                     device_image_to_host)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopyHostToMetalTexture,
                     host_to_device_image)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kNCHW))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kMetalTexture2DArray))})
.Finalize();


REGISTER_LITE_KERNEL(io_copy_once,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopykMetalTextureToHost,
                     device_image_to_host)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW))})
.Finalize();


REGISTER_LITE_KERNEL(io_copy,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopyHostToMetalTextureHalf,
                     host_to_device_image)
        .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopykMetalTextureToHostHalf,
                     device_image_to_host)
        .BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
        .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopyHostToMetalTextureHalf,
                     host_to_device_image)
        .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(io_copy_once,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::IoCopykMetalTextureToHostHalf,
                     device_image_to_host)
        .BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
        .Finalize();


