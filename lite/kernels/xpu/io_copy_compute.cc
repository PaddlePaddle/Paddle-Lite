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

#include "lite/backends/xpu/target_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

/*
 * This kernel copies a tensor from host to XPU.
 */
class IoCopyHostToXPUCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kX86) ||
          param.x->target() == TARGET(kARM));
    auto mem_size = param.x->memory_size();
    VLOG(4) << "host to xpu, copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kXPU), mem_size);
    TargetWrapperXPU::MemcpySync(
        data, param.x->raw_data(), mem_size, IoDirection::HtoD);
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->target() == TARGET(kHost));

      auto out_place = type->place();
      out_place.target = TARGET(kXPU);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to XPU"; }
};

/*
 * This kernel copies a tensor from XPU to host.
 */
class IoCopyXPUToHostCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kXPU));
    auto mem_size = param.x->memory_size();
    VLOG(4) << "xpu to host, copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kHost), mem_size);
    TargetWrapperXPU::MemcpySync(
        data, param.x->raw_data(), mem_size, IoDirection::DtoH);
  }

  std::string doc() const override { return "Copy IO from XPU to HOST"; }
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::IoCopyHostToXPUCompute,
                     host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::IoCopyXPUToHostCompute,
                     device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::IoCopyHostToXPUCompute,
                     host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::IoCopyXPUToHostCompute,
                     device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
