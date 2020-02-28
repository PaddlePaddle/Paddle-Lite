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

#include "lite/backends/opencl/cl_half.h"
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

class ActivationComputeImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ActivationParam;

  std::string doc() const override {
    return "Activation using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    act_param_ = param_.get_mutable<param_t>();
    int act_type = static_cast<int>(act_param_->active_type);
    switch (act_type) {
      case 1:
        kernel_func_name_ = "relu";
        break;
      case 2:
        kernel_func_name_ = "relu6";
        threshold_ = act_param_->Relu_clipped_coef;
        break;
      case 4:
        kernel_func_name_ = "leakyRelu";
        scale_ = act_param_->Leaky_relu_alpha;
        break;
      case 5:
        kernel_func_name_ = "sigmoid";
        break;
      case 6:
        kernel_func_name_ = "tanhAct";
        break;
      default:
        printf("This act type: %d doesn't support \n", act_type);
        return;
    }
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/activation_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_img = param.X->data<half_t, cl::Image2D>();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_img = param.Out->mutable_data<half_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);
    const auto& y_dims = param.Out->dims();  // useless: check dim only

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, threshold_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, scale_);
    CL_CHECK_FATAL(status);

    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "threshold:" << threshold_;
    VLOG(4) << "scale:" << scale_;
    VLOG(4) << "kernel func name:" << kernel_func_name_;

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
    context.cl_wait_list()->emplace(out_img, event_);
  }

 private:
  param_t* act_param_{nullptr};
  std::string kernel_func_name_{};
  float threshold_{6.f};
  float scale_{1.f};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class ReluComputeImageDefault : public KernelLite<TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ActivationParam;

  std::string doc() const override {
    return "Relu using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/activation_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_img = param.X->data<half_t, cl::Image2D>();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_img = param.Out->mutable_data<half_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);
    const auto& y_dims = param.Out->dims();  // useless: check dim only

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, 6.f);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, 1.f);
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
    context.cl_wait_list()->emplace(out_img, event_);
  }

 private:
  std::string kernel_func_name_{"relu"};
  std::string build_options_{"-DCL_DTYPE_half -DRELU"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class Relu6ComputeImageDefault : public KernelLite<TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ActivationParam;

  std::string doc() const override {
    return "Relu6 using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/activation_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_img = param.X->data<half_t, cl::Image2D>();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_img = param.Out->mutable_data<half_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);
    const auto& y_dims = param.Out->dims();  // useless: check dim only
    auto threshold = param.Relu_clipped_coef;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, threshold);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, 1.f);
    CL_CHECK_FATAL(status);

    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "threshold:" << threshold;

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
    context.cl_wait_list()->emplace(out_img, event_);
  }

 private:
  std::string kernel_func_name_{"relu6"};
  std::string build_options_{"-DCL_DTYPE_half -DRELU6"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class SigmoidComputeImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ActivationParam;

  std::string doc() const override {
    return "Sigmoid using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/activation_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto* x_img =
        param.X->data<half_t,
                      cl::Image2D>();  // use half_t represents half float
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_img = param.Out->mutable_data<half_t, cl::Image2D>(  // use half_t
        // represents half float
        image_shape["width"],
        image_shape["height"]);
    const auto& y_dims = param.Out->dims();  // useless: check dim only

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, 6.f);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, 1.f);
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
    context.cl_wait_list()->emplace(out_img, event_);
  }

 private:
  std::string kernel_func_name_{"sigmoid"};
  std::string build_options_{"-DCL_DTYPE_half -DSIGMOID"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
// activation
REGISTER_LITE_KERNEL(
    activation,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::ActivationComputeImageDefault,
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

// Relu
/*
REGISTER_LITE_KERNEL(relu,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ReluComputeImageDefault,
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
*/
// Relu6
REGISTER_LITE_KERNEL(relu6,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::Relu6ComputeImageDefault,
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

// Sigmoid
REGISTER_LITE_KERNEL(sigmoid,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SigmoidComputeImageDefault,
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
