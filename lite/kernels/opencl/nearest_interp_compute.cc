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

#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class NearestInterpComputeFloatImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFloat),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "NearestInterp using cl::Image2D(ImageDefault/RGBA), kFloat";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/nearest_interp_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_buf = param.X->data<float, cl::Image2D>();
    auto* out_buf =
        param.Out->mutable_data<float, cl::Image2D>(param.out_w, param.out_h);
    const auto& y_dims = param.Out->dims();  // useless: check dim only
    float scale_h = y_dims[2] / x_dims[2];
    float scale_w = y_dims[3] / x_dims[3];
    int in_dims_h = x_dims[2];
    int out_dims_h = y_dims[2];
    int in_dims_w = x_dims[3];
    int out_dims_w = y_dims[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_w));
    CL_CHECK_FATAL(status);

    paddle::lite::CLImageConverterDefault default_convertor;
    auto y_img_shape = default_convertor.InitImageDimInfoWith(y_dims);  // w, h
    auto y_img_width = y_img_shape[0];
    LOG(INFO) << "y_img_width:" << y_img_width;

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(y_img_width / y_dims[3]),
                    static_cast<cl::size_type>(y_dims[3]),
                    static_cast<cl::size_type>(y_dims[0] * y_dims[2])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    // TODO(ysh329): io_copy(device->host) jammed if emplace to `cl_wait_list`
    // context.cl_wait_list()->emplace(out_buf, event_);
    context.cl_context()->GetCommandQueue().finish();
  }

 private:
  std::string kernel_func_name_{"nearest_interp"};
  std::string build_options_{"-DCL_DTYPE_float "};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class NearestInterpComputeFP16ImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "NearestInterp using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/nearest_interp_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_buf =
        param.X->data<int16_t,
                      cl::Image2D>();  // use int16_t represents half float
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_buf =
        param.Out->mutable_data<int16_t, cl::Image2D>(  // use int16_t
                                                        // represents half float
            image_shape["width"],
            image_shape["height"]);
    const auto& y_dims = param.Out->dims();  // useless: check dim only
    float scale_h = y_dims[2] / x_dims[2];
    float scale_w = y_dims[3] / x_dims[3];
    int in_dims_h = x_dims[2];
    int out_dims_h = y_dims[2];
    int in_dims_w = x_dims[3];
    int out_dims_w = y_dims[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_w));
    CL_CHECK_FATAL(status);

    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(image_shape["width"]),
                    static_cast<cl::size_type>(image_shape["height"])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    // TODO(ysh329): io_copy(device->host) jammed if emplace to `cl_wait_list`
    // context.cl_wait_list()->emplace(out_buf, event_);
    context.cl_context()->GetCommandQueue().finish();
  }

 private:
  std::string kernel_func_name_{"nearest_interp"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    nearest_interp,
    kOpenCL,
    kFloat,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeFloatImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeFP16ImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
