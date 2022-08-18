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

#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/utils/timer.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// Host to OpenCL memory.
float CopyFromHostSync(void* target, const void* source, size_t size) {
#ifdef LITE_WITH_PROFILE
  lite::Timer timer;
  timer.Start();
#endif
  TargetWrapperCL::MemcpySync(target, source, size, IoDirection::HtoD);
#ifdef LITE_WITH_PROFILE
  auto h2d_duration = timer.Stop();
  return h2d_duration;
#else
  return 0.0;
#endif
}

// Device to Host memory.
float CopyToHostSync(void* target, const void* source, size_t size) {
#ifdef LITE_WITH_PROFILE
  lite::Timer timer;
  timer.Start();
#endif
  CLRuntime::Global()->command_queue().finish();
  TargetWrapperCL::MemcpySync(target, source, size, IoDirection::DtoH);
#ifdef LITE_WITH_PROFILE
  auto d2h_duration = timer.Stop();
  return d2h_duration;
#else
  return 0.0;
#endif
}
// Device to Device memory.
float CopyFromDeviceToDeviceSync(void* target,
                                 const void* source,
                                 size_t size) {
  CLRuntime::Global()->command_queue().finish();
#ifdef LITE_WITH_PROFILE
  lite::Timer timer;
  timer.Start();
#endif
  TargetWrapperCL::MemcpySync(target, source, size, IoDirection::DtoD);
#ifdef LITE_WITH_PROFILE
  auto d2h_duration = timer.Stop();
  return d2h_duration;
#else
  return 0.0;
#endif
}

/*
 * This kernel copies a tensor from host to OpenCL space.
 */
class IoCopyHostToOpenCLCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "HostToOpenCL";
    ch->io_duration = h2d_duration_;
  }
#endif

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x);
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kARM));

    auto mem_size = param.x->memory_size();
#ifdef LITE_WITH_LOG
    VLOG(2) << "param.x->memory_size():" << mem_size;
    VLOG(2) << "param.x->dims().size():" << param.x->dims().size();
    VLOG(2) << "param.x->dims():" << param.x->dims();
    VLOG(2) << "param.y->dims().size():" << param.y->dims().size();
    VLOG(2) << "param.y->dims():" << param.y->dims();
#endif
    auto* data = param.y->mutable_data(TARGET(kOpenCL), mem_size);
    CHECK(data);
    CHECK(param.x->raw_data());
    h2d_duration_ = CopyFromHostSync(data, param.x->raw_data(), mem_size);
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->target() == TARGET(kHost));

      auto out_place = type->place();
      out_place.target = TARGET(kOpenCL);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to OpenCL"; }

  float h2d_duration_{0};
};

/*
 * This kernel copies a tensor from OpenCL to host space.
 */
class IoCopykOpenCLToHostCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "OpenCLToHost";
    ch->io_duration = d2h_duration_;
  }
#endif

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kOpenCL));
    auto mem_size = param.x->memory_size();
    auto* data = param.y->mutable_data(TARGET(kHost), mem_size);
    const cl::Buffer* x_ptr;
    if (param.process_type == 1) {
      x_ptr = param.x->data<uint8_t, cl::Buffer>();
      param.y->set_precision(PRECISION(kUInt8));
    } else {
      x_ptr = param.x->data<float, cl::Buffer>();
      param.y->set_precision(PRECISION(kFloat));
    }

#ifdef LITE_WITH_LOG
    VLOG(4) << "copy size " << mem_size;
    VLOG(4) << "param.x->dims().size():" << param.x->dims().size();
    VLOG(4) << "param.x->dims():" << param.x->dims();
    VLOG(4) << "param.y->dims().size():" << param.y->dims().size();
    VLOG(4) << "param.y->dims():" << param.y->dims();
    VLOG(4) << "param.process_type:" << param.process_type;
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
#endif

    d2h_duration_ = CopyToHostSync(data, param.x->raw_data(), mem_size);
  }

  std::string doc() const override { return "Copy IO from OpenCL to HOST"; }

  float d2h_duration_{0};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopyHostToOpenCLCompute,
                     host_to_device)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopykOpenCLToHostCompute,
                     device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopyHostToOpenCLCompute,
                     host_to_device)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopykOpenCLToHostCompute,
                     device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

#define LITE_WITH_LOG
