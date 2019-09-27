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

#include <memory>
#include <string>
#include "lite/core/kernel.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class LayoutComputeBufferChwToImage2DHwc
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/layout_kernel.cl", build_options_);
  }

  DDim InitImageDimInfoWith(const DDim& tensor_dim) {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }
    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];
    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;
    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));
  }

  void Run() override {
    auto& param = Param<param_t>();
    auto* x_data = param.x->data<float, cl::Buffer>();
    auto x_dims = param.x->dims();
    // auto* y_data = param.y->mutable_data<float,
    // cl::Image2D>(TARGET(kOpenCL));
    // TODO(ysh329): image shape compute
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* y_data = param.y->mutable_data<float, cl::Image2D>(
        TARGET(kOpenCL), image_shape[0], image_shape[1]);
    auto y_dims = param.y->dims();

    VLOG(4) << "image_shape[0]:" << image_shape[0]
            << " image_shape[1]:" << image_shape[1]
            << " image_shape.size():" << image_shape.size();
    VLOG(4) << "param.x->dims().size():" << param.x->dims().size();
    VLOG(4) << "param.x->dims():" << param.x->dims()[0] << " "
            << param.x->dims()[1] << " " << param.x->dims()[2] << " "
            << param.x->dims()[3];
    VLOG(4) << "param.y->dims().size():" << param.y->dims().size();
    VLOG(4) << "param.y->dims():" << param.y->dims()[0] << " "
            << param.y->dims()[1] << " " << param.y->dims()[2] << " "
            << param.y->dims()[3];
    // param.x->image2d_shape()
    VLOG(4) << "TargetToStr(param.x->target()):"
            << TargetToStr(param.x->target());
    VLOG(4) << "TargetToStr(param.y->target()):"
            << TargetToStr(param.y->target());

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    // TODO(ysh329): impl kernel and checck args

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(image_shape[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(image_shape[0]));
    CL_CHECK_FATAL(status);
    // TODO(ysh329): global work size setting
    size_t gws_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < x_dims.size(); ++j) {
      gws_dims[4 - x_dims.size() + j] = x_dims[j];
    }

    VLOG(4) << "gws_dims[0]" << gws_dims[0] << " "
            << "gws_dims[1]" << gws_dims[1] << " "
            << "gws_dims[2]" << gws_dims[2] << " "
            << "gws_dims[3]" << gws_dims[3] << " ";
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(gws_dims[1]),
                    static_cast<cl::size_type>(gws_dims[2]),
                    static_cast<cl::size_type>(gws_dims[3])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(y_data, event_);
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);

    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->layout() == DATALAYOUT(kNCHW));

      auto out_place = type->place();
      out_place.layout = DATALAYOUT(kNHWC);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };

    return res;
  }

  std::string doc() const override { return "Trans Layout from NCHW to NHWC"; }

 private:
  std::string kernel_func_name_{"BufferChwToImgHwc"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class LayoutComputeImage2DHwcToBufferChw
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/layout_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    auto* x_data = param.x->data<float, cl::Image2D>();
    // const cl::Image2D* x_data = param.x->image_data();
    auto x_dims = param.x->dims();
    auto* y_data = param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto y_dims = param.y->dims();
    VLOG(4) << "TargetToStr(param.x->target()):"
            << TargetToStr(param.x->target());
    VLOG(4) << "TargetToStr(param.y->target()):"
            << TargetToStr(param.y->target());
    VLOG(4) << "param.x->dims().size():" << param.x->dims().size();
    VLOG(4) << "param.x->dims():" << param.x->dims()[0] << " "
            << param.x->dims()[1] << " " << param.x->dims()[2] << " "
            << param.x->dims()[3];
    VLOG(4) << "param.y->dims().size():" << param.y->dims().size();
    VLOG(4) << "param.y->dims():" << param.y->dims()[0] << " "
            << param.y->dims()[1] << " " << param.y->dims()[2] << " "
            << param.y->dims()[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    // TODO(ysh329): impl kernel and checck args
    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[3]));
    CL_CHECK_FATAL(status);
    // TODO(ysh329): global work size setting
    for (int y_idx = y_dims.size() - 1; y_idx < y_dims.size(); ++y_idx) {
      y_dims[y_idx] = 1;
    }
    auto global_work_size = cl::NDRange{static_cast<cl::size_type>(y_dims[1]),
                                        static_cast<cl::size_type>(y_dims[2]),
                                        static_cast<cl::size_type>(y_dims[3])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(y_data, event_);
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);

    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->layout() == DATALAYOUT(kNHWC));

      auto out_place = type->place();
      out_place.layout = DATALAYOUT(kNCHW);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };

    return res;
  }

  std::string doc() const override { return "Trans Layout from NHWC to NCHW"; }

 private:
  std::string kernel_func_name_{"ImgHwcToBufferChw"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// BufferChwToImage2DHwc
// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// Image2DHwcBufferChw
// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
