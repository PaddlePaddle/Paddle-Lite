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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

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
    act_param_ = param_.get_mutable<param_t>();
    act_type_ = static_cast<int>(act_param_->active_type);
#ifdef LITE_WITH_LOG
    VLOG(1) << "ActivationTypeToStr(act_param_->active_type):"
            << ActivationTypeToStr(act_param_->active_type);
#endif
    switch (act_type_) {
      case 1:
        kernel_func_name_ = "relu";
        break;
      case 2:
        kernel_func_name_ = "relu6";
        threshold_ = act_param_->Relu_clipped_coef;
        break;
      case 3:
        mode_ = act_param_->Prelu_mode;
        if (mode_ == "all") {
          kernel_func_name_ = "leaky_relu";
          const float* alpha_data = act_param_->Prelu_alpha->data<float>();
          scale_ = alpha_data[0];
        } else if (mode_ == "channel") {
          alpha_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
          tensor_hold_alpha_image_ = std::unique_ptr<Tensor>(new Tensor);
          kernel_func_name_ = "prelu_channel";
          auto& out_dims = act_param_->Out->dims();
          if (out_dims.size() == 4) {
            width_ = out_dims[3];
            CLImageConverterFolder alpha_converter;
            const DDim& alpha_image_dims = alpha_converter.InitImageDimInfoWith(
                act_param_->Prelu_alpha->dims());
            tensor_hold_alpha_image_->Resize(
                {1, alpha_image_dims[0], alpha_image_dims[1], 4});

            auto* alpha_image_data = MUTABLE_DATA_CPU(tensor_hold_alpha_image_);
            auto* alpha_cpu_data =
                act_param_->Prelu_alpha->mutable_data<float>();
            alpha_converter.NCHWToImage(alpha_cpu_data,
                                        alpha_image_data,
                                        act_param_->Prelu_alpha->dims());

            MUTABLE_DATA_GPU(alpha_gpu_image_,
                             alpha_image_dims[0],
                             alpha_image_dims[1],
                             alpha_image_data);
          } else if (out_dims.size() == 2) {
            width_ = 1;
            CLImageConverterDefault alpha_converter;
            const DDim& alpha_image_dims = alpha_converter.InitImageDimInfoWith(
                act_param_->Prelu_alpha->dims());
            tensor_hold_alpha_image_->Resize(
                {1, alpha_image_dims[0], alpha_image_dims[1], 4});

            auto* alpha_image_data = MUTABLE_DATA_CPU(tensor_hold_alpha_image_);
            auto* alpha_cpu_data =
                act_param_->Prelu_alpha->mutable_data<float>();
            alpha_converter.NCHWToImage(alpha_cpu_data,
                                        alpha_image_data,
                                        act_param_->Prelu_alpha->dims());

            MUTABLE_DATA_GPU(alpha_gpu_image_,
                             alpha_image_dims[0],
                             alpha_image_dims[1],
                             alpha_image_data);
          } else {
            LOG(FATAL) << "unsupport dims.size(): " << out_dims.size();
          }
        } else {
          alpha_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
          tensor_hold_alpha_image_ = std::unique_ptr<Tensor>(new Tensor);
          kernel_func_name_ = "prelu_element";
          auto& in_dim = act_param_->X->dims();
          if (in_dim.size() > 3) {
            height_ = in_dim[2];
          } else {
            height_ = 1;
          }
          scale_ = act_param_->Leaky_relu_alpha;

          CLImageConverterDefault alpha_converter;
          const DDim& alpha_image_dims = alpha_converter.InitImageDimInfoWith(
              act_param_->Prelu_alpha->dims());
          tensor_hold_alpha_image_->Resize(
              {1, alpha_image_dims[0], alpha_image_dims[1], 4});

          auto* alpha_image_data = MUTABLE_DATA_CPU(tensor_hold_alpha_image_);
          auto* alpha_cpu_data = act_param_->Prelu_alpha->mutable_data<float>();
          alpha_converter.NCHWToImage(alpha_cpu_data,
                                      alpha_image_data,
                                      act_param_->Prelu_alpha->dims());

          MUTABLE_DATA_GPU(alpha_gpu_image_,
                           alpha_image_dims[0],
                           alpha_image_dims[1],
                           alpha_image_data);
        }
        break;
      case 4:
        kernel_func_name_ = "leaky_relu";
        scale_ = act_param_->Leaky_relu_alpha;
        break;
      case 5:
        kernel_func_name_ = "sigmoid";
        break;
      case 6:
        kernel_func_name_ = "tanh_act";
        break;
      case 7:
        kernel_func_name_ = "swish";
        scale_ = act_param_->Swish_beta;
        break;
      case 8:
        kernel_func_name_ = "exp_act";
        break;
      case 9:
        kernel_func_name_ = "abs_act";
        break;
      case 10:
        kernel_func_name_ = "hard_swish";
        scale_ = act_param_->hard_swish_scale;
        threshold_ = act_param_->hard_swish_threshold;
        offset_ = act_param_->hard_swish_offset;
        break;
      case 14:
        kernel_func_name_ = "hard_sigmoid";
        scale_ = act_param_->hard_sigmoid_slope;
        threshold_ = act_param_->hard_sigmoid_offset;
        break;

      case 15:
        kernel_func_name_ = "log_act";
        break;
      case 18:
        kernel_func_name_ = "gelu";

        break;
      default:
        LOG(FATAL) << "This act type:" << act_type_ << " doesn't support.";
        return;
    }
#ifdef LITE_WITH_LOG
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/activation_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    act_param_ = param_.get_mutable<param_t>();
    auto x_dims = act_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      x_img_shape_ = default_convertor.InitImageDimInfoWith(
          act_param_->X->dims());  // w, h
      out_img_shape_ = default_convertor.InitImageDimInfoWith(
          act_param_->Out->dims());  // w, h

      // compute global work size
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(x_img_shape_[0]),
                    static_cast<cl::size_type>(x_img_shape_[1])};
  }

  void Run() override {
    auto* x_img = GET_DATA_GPU(act_param_->X);
    auto* out_img = MUTABLE_DATA_GPU(
        act_param_->Out, out_img_shape_[0], out_img_shape_[1], nullptr);
    auto kernel = kernel_;
    int cnt = 2;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);
    if (kernel_func_name_ == "hard_sigmoid") {
      status = kernel.setArg(cnt++, threshold_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(cnt++, scale_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "leaky_relu" ||
               kernel_func_name_ == "swish") {
      status = kernel.setArg(cnt++, scale_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "prelu_channel") {
      auto* alpha_img_in = GET_DATA_GPU(alpha_gpu_image_);
      status = kernel.setArg(cnt++, width_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(cnt++, *alpha_img_in);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "prelu_element") {
      auto* alpha_img_in = GET_DATA_GPU(alpha_gpu_image_);
      status = kernel.setArg(cnt++, height_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(cnt++, *alpha_img_in);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "hard_swish") {
      status = kernel.setArg(cnt++, threshold_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(cnt++, scale_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(cnt++, offset_);
      CL_CHECK_FATAL(status);
    }

#ifdef LITE_WITH_LOG
    const auto& x_dims = act_param_->X->dims();
    const auto& y_dims = act_param_->Out->dims();  // useless: check dim only
    VLOG(4) << TargetToStr(act_param_->X->target());
    VLOG(4) << TargetToStr(act_param_->Out->target());
    VLOG(4) << "x_img_shape_(w,h):" << x_img_shape_[0] << " "
            << x_img_shape_[1];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "threshold:" << threshold_;
    VLOG(4) << "scale:" << scale_;
    VLOG(4) << "kernel func name:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }
#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 protected:
  param_t* act_param_{nullptr};
  DDim x_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim last_x_dims_;
  std::string kernel_func_name_{};
  float threshold_{6.f};
  float scale_{1.f};
  float offset_{3.f};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  std::string mode_{"channel"};
  int act_type_{0};
  int width_{0};
  int height_{0};
  std::unique_ptr<Tensor> alpha_gpu_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_alpha_image_{nullptr};
};

class SqrtComputeImageDefault : public ActivationComputeImageDefault {
  std::string doc() const override {
    return "Sqrt using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    kernel_func_name_ = "sqrt_func";
#ifdef LITE_WITH_LOG
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/activation_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void Run() override {
    auto* x_img = GET_DATA_GPU(act_param_->X);
    auto* out_img = MUTABLE_DATA_GPU(
        act_param_->Out, out_img_shape_[0], out_img_shape_[1], nullptr);
    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
    const auto& x_dims = act_param_->X->dims();
    const auto& y_dims = act_param_->Out->dims();  // useless: check dim only
    VLOG(4) << TargetToStr(act_param_->X->target());
    VLOG(4) << TargetToStr(act_param_->Out->target());
    VLOG(4) << "x_img_shape_(w,h):" << x_img_shape_[0] << " "
            << x_img_shape_[1];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "kernel func name:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }
};

class SquareComputeImageDefault : public SqrtComputeImageDefault {
  std::string doc() const override {
    return "Square using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    act_param_ = param_.get_mutable<param_t>();
    act_type_ = static_cast<int>(act_param_->active_type);
    kernel_func_name_ = "square_func";
#ifdef LITE_WITH_LOG
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/activation_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }
};

class RsqrtComputeImageDefault : public SqrtComputeImageDefault {
  std::string doc() const override {
    return "Rsqrt using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    act_param_ = param_.get_mutable<param_t>();
    act_type_ = static_cast<int>(act_param_->active_type);
    kernel_func_name_ = "rsqrt_func";
#ifdef LITE_WITH_LOG
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/activation_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// leakyRelu
REGISTER_LITE_KERNEL(
    leaky_relu,
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

// swish
REGISTER_LITE_KERNEL(
    swish,
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

// exp
REGISTER_LITE_KERNEL(
    exp,
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

// tanh
REGISTER_LITE_KERNEL(
    tanh,
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
REGISTER_LITE_KERNEL(
    relu,
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

// abs
REGISTER_LITE_KERNEL(
    abs,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::ActivationComputeImageDefault,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// Relu6
REGISTER_LITE_KERNEL(
    relu6,
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

// Sigmoid
REGISTER_LITE_KERNEL(
    sigmoid,
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

// Hard Sigmoid
REGISTER_LITE_KERNEL(
    hard_sigmoid,
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

// Hard Swish
REGISTER_LITE_KERNEL(
    hard_swish,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::ActivationComputeImageDefault,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// Gelu
REGISTER_LITE_KERNEL(
    gelu,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::ActivationComputeImageDefault,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// Prelu
REGISTER_LITE_KERNEL(
    prelu,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::ActivationComputeImageDefault,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("mode", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// sqrt
REGISTER_LITE_KERNEL(sqrt,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SqrtComputeImageDefault,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// rsqrt
REGISTER_LITE_KERNEL(rsqrt,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::RsqrtComputeImageDefault,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// square
REGISTER_LITE_KERNEL(square,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SquareComputeImageDefault,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(
    log,
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