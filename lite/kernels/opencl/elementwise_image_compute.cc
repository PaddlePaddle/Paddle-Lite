// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <memory>
#include <string>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/host/elementwise_op_func.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ElementwiseImageCompute : public KernelLite<TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ElementwiseParam;

  std::string doc() const override {
    return "Elementwise using cl::Image2D(ImageDefault/RGBA), kFP32";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    if (param_.is_type<param_t>()) {
      ele_param_ = param_.get_mutable<param_t>();
    } else {
      ele_param_ =
          param_.get_mutable<operators::FusionElementwiseActivationParam>();
      auto act_t =
          static_cast<operators::FusionElementwiseActivationParam*>(ele_param_)
              ->act_type;
      VLOG(4) << "act: " << act_t;
      if (act_t == "relu") {
        build_options_ += " -DRELU";
      } else if (act_t == "relu6") {
        build_options_ += " -DRELU6";
      } else if (act_t == "gelu") {
        build_options_ += " -DGELU";
      } else {
        LOG(FATAL) << "Unsupported Activation type: " << act_t;
      }
    }

    auto* x = ele_param_->X;
    auto* y = ele_param_->Y;
    x_dims_ = ele_param_->X->dims();
    y_dims_ = ele_param_->Y->dims();
    auto& out_dims = ele_param_->Out->dims();
    axis_ = ele_param_->axis;
    out_nchw_ = out_dims.Vectorize();

    if (x_dims_ == y_dims_) {
      axis_ = -1;
    }
    host::fix_x_y_dims<int64_t>(
        x, y, ele_param_->Out, axis_, &x_nchw_, &y_nchw_);

    while (x_nchw_.size() < 4) {
      x_nchw_.insert(x_nchw_.cbegin(), 1);
    }

    while (y_nchw_.size() < 4) {
      y_nchw_.insert(y_nchw_.cbegin(), 1);
    }

    while (out_nchw_.size() < 4) {
      out_nchw_.insert(out_nchw_.cbegin(), 1);
    }

    image_folder_flag_y_ = 0;
    image_folder_flag_x_ = 0;
    int broadcast_elementwise_common_flag = 0;
    if (y->persistable()) {
      LOG(INFO) << "with y->persistable";
      y_weights_image_ = std::unique_ptr<Tensor>(new Tensor);
      std::unique_ptr<Tensor> tensor_hold_y_image_ =
          std::unique_ptr<Tensor>(new Tensor);
      CLImageConverterDefault default_converter;
      const DDim& y_image_dims =
          default_converter.InitImageDimInfoWith(DDim(y_nchw_));
      tensor_hold_y_image_->Resize({1, y_image_dims[0], y_image_dims[1], 4});

      auto* y_cpu_image = MUTABLE_DATA_CPU(tensor_hold_y_image_);
      auto* y_cpu_nchw = static_cast<float*>(const_cast<void*>(y->raw_data()));
      default_converter.NCHWToImage(y_cpu_nchw, y_cpu_image, DDim(y_nchw_));
      MUTABLE_DATA_GPU(
          y_weights_image_, y_image_dims[0], y_image_dims[1], y_cpu_image);
    } else {
      if ((y_dims_.size() == 1) &&
          (axis_ != -1) &&  // x{n,c,h,w} && y{c} || x{c,h,w} && y{c}
          (axis_ == x_dims_.size() - 3)) {
        image_folder_flag_y_ = 1;
      }
      if ((y_dims_.size() == 1) &&
          (axis_ != -1) &&  // x{n,c,h,w} && y{h} || x{c,h,w} && y{h}
          (axis_ == x_dims_.size() - 2 || axis_ == x_dims_.size() - 4)) {
        image_folder_flag_y_ = 2;
      }
      if ((y_dims_.size() == 2) &&
          (axis_ != -1) &&  // x{n,c,h,w} && y{c,h} || x{c,h,w} && y{c,h}
          (axis_ == x_dims_.size() - 3)) {
        image_folder_flag_y_ = 3;
      }
      if ((y_dims_.size() == 2) && (x_dims_.size() == 4) &&
          (axis_ == 0)) {  // x{n,c,h,w} && y{n,c}
        image_folder_flag_y_ = 1;
      }
      if ((y_dims_.size() == 3) && (x_dims_.size() == 4) &&
          (axis_ == 0)) {  // x{n,c,h,w} && y{n,c,h}
        image_folder_flag_y_ = 4;
      }

      if ((x_dims_.size() == 1) && (axis_ != -1) &&
          (axis_ == y_dims_.size() - 3)) {
        image_folder_flag_x_ = 1;
      }
      if ((x_dims_.size() == 1) && (axis_ != -1) &&
          (axis_ == y_dims_.size() - 2 || axis_ == y_dims_.size() - 4)) {
        image_folder_flag_x_ = 2;
      }
      if ((x_dims_.size() == 2) && (axis_ != -1) &&
          (axis_ == y_dims_.size() - 3)) {
        image_folder_flag_x_ = 3;
      }
      if ((x_dims_.size() == 2) && (y_dims_.size() == 4) && (axis_ == 0)) {
        image_folder_flag_x_ = 1;
      }
      if ((x_dims_.size() == 3) && (y_dims_.size() == 4) && (axis_ == 0)) {
        image_folder_flag_x_ = 4;
      }
    }

    if (image_folder_flag_x_ != 0 || image_folder_flag_y_ != 0) {
      broadcast_elementwise_common_flag = 1;
    }

    if (y_dims_ == x_dims_) {
      kernel_func_name_ = "elementwise_compute";
      kernel_func_paths_ = "image/elementwise_kernel.cl";
    } else if (broadcast_elementwise_common_flag == 0) {
      kernel_func_name_ = "broadcast_elementwise_basic";
      kernel_func_paths_ = "image/elementwise_kernel.cl";
    } else {
      kernel_func_name_ = "broadcast_elementwise_common";
      kernel_func_paths_ = "image/elementwise_broadcast_kernel.cl";
    }

    // op_type
    auto elementwise_compute_type = op_type();
    if (elementwise_compute_type == "elementwise_div" ||
        elementwise_compute_type == "fusion_elementwise_div_activation") {
      build_options_ += " -DOPERATOR(in,bias)=(in/bias) ";
    } else if (elementwise_compute_type == "elementwise_add" ||
               elementwise_compute_type ==
                   "fusion_elementwise_add_activation") {
      build_options_ += " -DOPERATOR(in,bias)=(in+bias) ";
    } else if (elementwise_compute_type == "elementwise_sub" ||
               elementwise_compute_type ==
                   "fusion_elementwise_sub_activation") {
      build_options_ += " -DOPERATOR(in,bias)=(in-bias) ";
    } else if (elementwise_compute_type == "elementwise_mul" ||
               elementwise_compute_type ==
                   "fusion_elementwise_mul_activation") {
      build_options_ += " -DOPERATOR(in,bias)=(in*bias) ";
    } else if (elementwise_compute_type == "elementwise_max") {
      build_options_ += " -DOPERATOR(in,bias)=fmax(in,bias) ";
    } else if (elementwise_compute_type == "elementwise_min") {
      build_options_ += " -DOPERATOR(in,bias)=fmin(in,bias) ";
    } else if (elementwise_compute_type == "elementwise_pow") {
      build_options_ += " -DOPERATOR(in,bias)=pow(in,bias) ";
    } else if (elementwise_compute_type == "elementwise_mod") {
      build_options_ += " -DOPERATOR(in,bias)=fmod(in,bias) ";
    } else if (elementwise_compute_type == "elementwise_floordiv") {
      build_options_ += " -DOPERATOR(in,bias)=(int4)(in/bias) ";
    }

    if (ele_param_->fuse_scale) {
      build_options_ +=
          "-DFUSE_SCALE -DSCALE_SLOPE=" + std::to_string(ele_param_->scale) +
          "f " + " -DSCALE_BIAS=" + std::to_string(ele_param_->bias) + "f " +
          " -DSCALE_ALPHA=" + std::to_string(ele_param_->alpha) + "f ";
    }
    context.cl_context()->AddKernel(
        kernel_func_name_, kernel_func_paths_, build_options_, time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    std::string fuse_scale_str = ele_param_->fuse_scale ? "/fuse_scale" : "";
    ch->kernel_func_name = kernel_func_name_ + fuse_scale_str;
    ch->global_work_size = ch->NDRangeToStr(gws_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void ReInitWhenNeeded() override {
    if ((!first_epoch_for_reinit_ && x_dims_ != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims_;
      first_epoch_for_reinit_ = false;

      // compute global work size
      int hb = out_nchw_[0] * out_nchw_[2];
      int cw =
          out_nchw_[3] *
          maptofactor(out_nchw_[1], 4);  // return (i + factor - 1) / factor;

      gws_ = cl::NDRange{static_cast<cl::size_type>(cw),
                         static_cast<cl::size_type>(hb),
                         static_cast<cl::size_type>(1)};
    }
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    const auto* x_img = GET_DATA_GPU(ele_param_->X);
    const auto* y_img = GET_DATA_GPU(ele_param_->Y);
    if (ele_param_->Y->persistable()) {
      y_img = GET_DATA_GPU(y_weights_image_);
    }
    auto out_image_shape = InitImageDimInfoWith(DDim(out_nchw_));  // w, h
    auto* out_img = MUTABLE_DATA_GPU(ele_param_->Out,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);

    // nchw --> nhwc4
    cl_int4 inx_dim = {static_cast<int>(x_nchw_[0]),
                       static_cast<int>(x_nchw_[2]),
                       static_cast<int>(x_nchw_[3]),
                       static_cast<int>((x_nchw_[1] + 3) / 4)};

    cl_int4 iny_dim = {static_cast<int>(y_nchw_[0]),
                       static_cast<int>(y_nchw_[2]),
                       static_cast<int>(y_nchw_[3]),
                       static_cast<int>((y_nchw_[1] + 3) / 4)};

    cl_int4 out_dim = {static_cast<int>(out_nchw_[0]),
                       static_cast<int>(out_nchw_[2]),
                       static_cast<int>(out_nchw_[3]),
                       static_cast<int>((out_nchw_[1] + 3) / 4)};

    int inputx_broadcast_c_flag = (x_nchw_[1] == 1) ? 1 : 0;
    int inputy_broadcast_c_flag = (y_nchw_[1] == 1) ? 1 : 0;
    int bias_width = out_nchw_[1];

    if (y_dims_ == x_dims_) {
      cl_int status = kernel_.setArg(0, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(1, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(2, *out_img);
      CL_CHECK_FATAL(status);
    } else {
      cl_int status = kernel_.setArg(0, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(1, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(2, *out_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(3, inx_dim);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(4, iny_dim);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(5, out_dim);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(6, inputx_broadcast_c_flag);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(7, inputy_broadcast_c_flag);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(8, image_folder_flag_x_);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(9, image_folder_flag_y_);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(10, bias_width);
      CL_CHECK_FATAL(status);
    }

    auto status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_PROFILE
    event_.wait();
    auto queue_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    auto submit_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    auto run_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
    VLOG(4) << "GetQueuedToSubmitTime: " << time_ms << std::endl;

    time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
    VLOG(4) << "GetSubmitToStartTime: " << time_ms << std::endl;

    time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
    VLOG(4) << "GetStartToEndTime: " << time_ms << std::endl;
#endif
  }

 private:
  param_t* ele_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::vector<int64_t> x_nchw_{};
  std::vector<int64_t> y_nchw_{};
  std::vector<int64_t> out_nchw_{};
  std::string kernel_func_name_{};
  std::string build_options_{};
  std::string kernel_func_paths_{};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_;
  // y is persistable
  std::unique_ptr<Tensor> y_weights_image_{nullptr};
  int image_folder_flag_x_{0};
  int image_folder_flag_y_{0};
  int axis_{-1};
  DDimLite x_dims_{};
  DDimLite y_dims_{};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(elementwise_div,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_max,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_min,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_pow,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mod,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_elementwise_sub_activation,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_elementwise_mul_activation,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_elementwise_div_activation,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
REGISTER_LITE_KERNEL(elementwise_floordiv,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
