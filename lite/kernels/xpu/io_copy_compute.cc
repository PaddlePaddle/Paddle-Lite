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

// xpu runtime support pinned memory from 4.0.20 in xpu2
// if buffersize > 1MB, we use pinned mem to improve DMA performace,
#define PINNED_MEM_SIZE (1024 * 1024)

static bool check_pin_mem_condition() {
  int cur_dev_idx = 0;
  uint64_t cur_dev_attr = 0;
  XPU_CALL(xpu_current_device(&cur_dev_idx));
  XPU_CALL(xpu_device_get_attr(&cur_dev_attr, XPUATTR_MODEL, cur_dev_idx));
  if (cur_dev_attr <= 1) {
    VLOG(4) << "Currents XPU device : XPU1";
  } else if (cur_dev_attr >= 2 && cur_dev_attr <= 299) {
    VLOG(4) << "Currents XPU device : XPU2";
  } else if (cur_dev_attr >= 300 && cur_dev_attr <= 599) {
    VLOG(4) << "Currents XPU device : XPU3";
  } else {
    VLOG(4) << "invaid XPU device";
  }
  // get driver version
  uint32_t major = 0;
  uint32_t minor = 0;
  XPU_CALL(xpu_get_driver_version(&major, &minor));
  return (cur_dev_attr >= 2 && major >= 4);
}
/*
 * This kernel copies a tensor from host to XPU.
 */
class IoCopyHostToXPUCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void IoCopyHostToDevice(const Tensor* x, Tensor* y) {
    if (x->target() == TARGET(kHost) || x->target() == TARGET(kX86) ||
        x->target() == TARGET(kARM)) {
      auto mem_size = x->memory_size();
      VLOG(4) << "host to xpu, copy size " << mem_size;
      auto* data = y->mutable_data(TARGET(kXPU), mem_size);
      if (mem_size > 0) {
        if (can_pinned_ && this->op_type_ == "io_copy" &&
            mem_size > PINNED_MEM_SIZE) {
          x->set_host_pinned_memory();
        }
        TargetWrapperXPU::MemcpySync(
            data, x->raw_data(), mem_size, IoDirection::HtoD);
      }
    } else if (x->target() == TARGET(kXPU)) {
      y->ShareDataWith(*x);
    } else {
      LOG(FATAL) << "IoCopyHostToXPU can not handle with the input target: "
                 << lite_api::TargetToStr(x->target());
    }
  }

  void PrepareForRun() { can_pinned_ = check_pin_mem_condition(); }

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    if (param.x != nullptr) {
      IoCopyHostToDevice(param.x, param.y);
    }
    if (param.x_array != nullptr) {
      for (size_t i = 0; i < param.x_array->size(); i++) {
        IoCopyHostToDevice(&(param.x_array->at(i)), &(param.y_array->at(i)));
      }
    }
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      // CHECK(type->target() == TARGET(kHost));

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

 private:
  bool can_pinned_{false};
};

/*
 * This kernel copies a tensor from XPU to host.
 */
class IoCopyXPUToHostCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void IoCopyDeviceToHost(const Tensor* x, Tensor* y) {
    if (x->target() == TARGET(kXPU)) {
      auto mem_size = x->memory_size();
      VLOG(4) << "xpu to host, copy size " << mem_size;
      auto* data = y->mutable_data(TARGET(kHost), mem_size);
      if (mem_size > 0) {
        if (can_pinned_ && this->op_type_ == "io_copy" &&
            mem_size > PINNED_MEM_SIZE) {
          y->set_host_pinned_memory();
        }
        TargetWrapperXPU::MemcpySync(
            data, x->raw_data(), mem_size, IoDirection::DtoH);
      }
    } else if (x->target() == TARGET(kHost) || x->target() == TARGET(kX86) ||
               x->target() == TARGET(kARM)) {
      y->CopyDataFrom(*x);
    } else {
      LOG(FATAL) << "IoCopyXPUToHost can not handle with the input target: "
                 << lite_api::TargetToStr(x->target());
    }
  }

  void PrepareForRun() { can_pinned_ = check_pin_mem_condition(); }

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    if (param.x != nullptr) {
      IoCopyDeviceToHost(param.x, param.y);
    }
    if (param.x_array != nullptr) {
      for (size_t i = 0; i < param.x_array->size(); i++) {
        IoCopyDeviceToHost(&(param.x_array->at(i)), &(param.y_array->at(i)));
      }
    }
  }

  std::string doc() const override { return "Copy IO from XPU to HOST"; }

 private:
  bool can_pinned_{false};
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
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kXPU),
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
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kXPU),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kHost),
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
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kXPU),
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
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kXPU),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();
