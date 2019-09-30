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

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// Host to OpenCL memory.
void CopyFromHostSync(void* target, const void* source, size_t size) {
  TargetWrapperCL::MemcpySync(target, source, size, IoDirection::HtoD);
}

// Device to Host memory.
void CopyToHostSync(void* target, const void* source, size_t size) {
  TargetWrapperCL::MemcpySync(target, source, size, IoDirection::DtoH);
}

/*
 * This kernel copies a tensor from host to OpenCL space.
 */
class IoCopyHostToOpenCLCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kARM));
    auto mem_size = param.x->memory_size();
    VLOG(4) << "copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kOpenCL), mem_size);
    CopyFromHostSync(data, param.x->raw_data(), mem_size);
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
};

/*
 * This kernel copies a tensor from OpenCL to host space.
 */
class IoCopykOpenCLToHostCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kOpenCL));
    auto mem_size = param.x->memory_size();
    VLOG(4) << "copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kHost), mem_size);
    auto& context = ctx_->As<OpenCLContext>();
    auto* wait_list = context.cl_wait_list();
    auto* x_ptr = param.x->data<float, cl::Buffer>();
    auto it = wait_list->find(x_ptr);
    if (it != wait_list->end()) {
      VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
      auto& event = *(it->second);
      event.wait();
    } else {
      LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
    }
    CopyToHostSync(data, param.x->raw_data(), mem_size);
  }

  std::string doc() const override { return "Copy IO from OpenCL to HOST"; }
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
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopykOpenCLToHostCompute,
                     device_to_host)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopyHostToOpenCLCompute,
                     host_to_device)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::IoCopykOpenCLToHostCompute,
                     device_to_host)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
