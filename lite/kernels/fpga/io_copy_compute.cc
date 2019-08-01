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

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void CopyFromHostSync(void* target, const void* source, size_t size) {
  LOG(INFO) << "copying " << target << ":" << source << ":" << size;
  TargetWrapper<TARGET(kFPGA)>::MemcpySync(
      target, source, size, IoDirection::HtoD);
}

void CopyToHostSync(void* target, const void* source, size_t size) {
  TargetWrapper<TARGET(kFPGA)>::MemcpySync(
      target, source, size, IoDirection::DtoH);
}

/*
 * This kernel copies a tensor from host to FPGA space.
 */
class IoCopyHostToFpgaCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));
    auto mem_size = param.x->memory_size();
    LOG(INFO) << "copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kFPGA), mem_size);
    LOG(INFO) << "size:" << param.y->memory_size() << ":"
              << param.x->memory_size();
    LOG(INFO) << "get_data: " << data << ":" << param.x->raw_data();
    CopyFromHostSync(data, param.x->raw_data(), mem_size);
    LOG(INFO) << "copy_from_host ";
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->target() == TARGET(kHost));

      auto out_place = type->place();
      out_place.target = TARGET(kFPGA);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to FPGA"; }
};

/*
 * This kernel copies a tensor from FPGA to host space.
 */
class IoCopyFpgaToHostCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kFPGA));
    auto mem_size = param.x->memory_size();
    VLOG(4) << "copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kHost), mem_size);
    CopyToHostSync(data, param.x->raw_data(), mem_size);
  }

  std::string doc() const override { return "Copy IO from FPGA to HOST"; }
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostToFpgaCompute,
                     host_to_device_float_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCompute,
                     device_to_host_float_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostToFpgaCompute,
                     host_to_device_float_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCompute,
                     device_to_host_float_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCompute,
                     device_to_host_fp16_hwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
