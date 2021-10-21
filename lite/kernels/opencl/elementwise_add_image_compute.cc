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

#include "lite/kernels/opencl/elementwise_add_image_compute.h"
#include <memory>
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/core/op_registry.h"
#include "lite/utils/replace_stl/stream.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ElementwiseAddImageCompute::PrepareForRun() {
  if (param_.is_type<param_t>()) {
    ele_param_ = param_.get_mutable<param_t>();
  } else {
    ele_param_ =
        param_.get_mutable<operators::FusionElementwiseActivationParam>();
    auto act_t =
        static_cast<operators::FusionElementwiseActivationParam*>(ele_param_)
            ->act_type;
    VLOG(4) << "act: " << act_t;
    if (act_t != "relu") {
      LOG(FATAL) << "Unsupported Activation type: " << act_t;
    }
    build_options_ += " -DRELU";
  }
  // choose kernel
  auto* x = ele_param_->X;
  auto* y = ele_param_->Y;
  auto* out = ele_param_->Out;
  auto axis = ele_param_->axis;

  if (y->dims().size() == 4) {
    kernel_func_name_ = "elementwise_add";  // y: ImageDefault
    if (y->dims()[0] == 1 && y->dims()[2] == 1 && y->dims()[3] == 1) {
      kernel_func_name_ = "elementwise_add_n1h1w1";
    }
    if ((axis == -1) && (y->dims()[3] > x->dims()[3])) {
      build_options_ += " -DBROADCAST";
    }
  } else if (y->dims().size() == 1) {
    if (axis == x->dims().size() - 1) {
      kernel_func_name_ = "width_add";  // y: ImageDefault
      if (y->persistable()) {
        y_weights_image_ = std::unique_ptr<Tensor>(new Tensor);
        std::unique_ptr<Tensor> tensor_hold_y_image_ =
            std::unique_ptr<Tensor>(new Tensor);
        CLImageConverterDefault default_converter;
        const DDim& y_image_dims =
            default_converter.InitImageDimInfoWith(y->dims());
        tensor_hold_y_image_->Resize({1, y_image_dims[0], y_image_dims[1], 4});

        auto* y_cpu_image = MUTABLE_DATA_CPU(tensor_hold_y_image_);
        auto* y_cpu_nchw =
            static_cast<float*>(const_cast<void*>(y->raw_data()));
        default_converter.NCHWToImage(y_cpu_nchw, y_cpu_image, y->dims());
        MUTABLE_DATA_GPU(
            y_weights_image_, y_image_dims[0], y_image_dims[1], y_cpu_image);
      }
    } else if (axis == x->dims().size() - 3) {
      kernel_func_name_ = "channel_add";  // y: ImageFolder
      if (y->persistable()) {
        y_weights_image_ = std::unique_ptr<Tensor>(new Tensor);
        std::unique_ptr<Tensor> tensor_hold_y_image_ =
            std::unique_ptr<Tensor>(new Tensor);
        CLImageConverterFolder folder_converter;
        const DDim& y_image_dims =
            folder_converter.InitImageDimInfoWith(y->dims());
        tensor_hold_y_image_->Resize({1, y_image_dims[0], y_image_dims[1], 4});

        auto* y_cpu_image = MUTABLE_DATA_CPU(tensor_hold_y_image_);
        auto* y_cpu_nchw =
            static_cast<float*>(const_cast<void*>(y->raw_data()));
        folder_converter.NCHWToImage(y_cpu_nchw, y_cpu_image, y->dims());

        MUTABLE_DATA_GPU(
            y_weights_image_, y_image_dims[0], y_image_dims[1], y_cpu_image);
      }
    } else if (axis == -1 && y->dims()[0] == 1) {
      kernel_func_name_ = "channel_add";  // for opt
      if (y->persistable()) {
        LOG(INFO) << "with y->persistable";
        y_weights_image_ = std::unique_ptr<Tensor>(new Tensor);
        std::unique_ptr<Tensor> tensor_hold_y_image_ =
            std::unique_ptr<Tensor>(new Tensor);
        CLImageConverterFolder folder_converter;
        const DDim& y_image_dims =
            folder_converter.InitImageDimInfoWith(y->dims());
        tensor_hold_y_image_->Resize({1, y_image_dims[0], y_image_dims[1], 4});

        auto* y_cpu_image = MUTABLE_DATA_CPU(tensor_hold_y_image_);
        auto* y_cpu_nchw =
            static_cast<float*>(const_cast<void*>(y->raw_data()));
        folder_converter.NCHWToImage(y_cpu_nchw, y_cpu_image, y->dims());

        MUTABLE_DATA_GPU(
            y_weights_image_, y_image_dims[0], y_image_dims[1], y_cpu_image);
      }
    } else {
      LOG(FATAL) << "ElementwiseAddImage doesn't support axis:" << axis
                 << ", x->dims().size():" << x->dims().size()
                 << ", y->dims.size():" << y->dims().size();
    }
  } else {
    LOG(FATAL) << "ElementwiseAddImage doesn't support axis:" << axis
               << ", x->dims().size():" << x->dims().size()
               << ", y->dims.size():" << y->dims().size();
  }
  VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(kernel_func_name_,
                                  "image/elementwise_add_kernel.cl",
                                  build_options_,
                                  time_stamp_);

  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());
}

void ElementwiseAddImageCompute::ReInitWhenNeeded() {
  auto* x = ele_param_->X;
  auto* y = ele_param_->Y;
  auto* out = ele_param_->Out;
  auto x_dims = ele_param_->X->dims();
  if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    first_epoch_for_reinit_ = false;
    // compute image shape
    paddle::lite::CLImageConverterDefault default_convertor;
    out_img_shape_ =
        default_convertor.InitImageDimInfoWith(out->dims());  // w, h

    // compute global work size
    GetGlobalWorkSize();
  }
}

void ElementwiseAddImageCompute::GetGlobalWorkSize() {
  global_work_size_ =
      cl::NDRange{static_cast<cl::size_type>(out_img_shape_[0]),
                  static_cast<cl::size_type>(out_img_shape_[1])};
#ifdef LITE_WITH_LOG
  VLOG(4) << "global_work_size:[2D]:" << out_img_shape_[0] << " "
          << out_img_shape_[1];
#endif
}

void ElementwiseAddImageCompute::Run() {
  auto* x = ele_param_->X;
  auto* y = ele_param_->Y;
  auto* out = ele_param_->Out;
  auto axis = ele_param_->axis;
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  auto* x_img = GET_DATA_GPU(x);
  auto* y_img = GET_DATA_GPU(y);
  auto* out_img =
      MUTABLE_DATA_GPU(out, out_img_shape_[0], out_img_shape_[1], nullptr);
  const int tensor_w = x_dims[x_dims.size() - 1];

#ifdef LITE_WITH_LOG
  VLOG(4) << "x->target():" << TargetToStr(x->target());
  VLOG(4) << "y->target():" << TargetToStr(y->target());
  VLOG(4) << "out->target():" << TargetToStr(out->target());
  VLOG(4) << "x->dims():" << x->dims();
  VLOG(4) << "y->dims():" << y->dims();
  VLOG(4) << "out->dims():" << out->dims();
  VLOG(4) << "axis:" << axis;
  VLOG(4) << "out_img_shape_[w,h]:" << out_img_shape_[0] << " "
          << out_img_shape_[1];
#endif

  cl_int status;
  auto kernel = kernel_;
  if (kernel_func_name_ == "elementwise_add") {
    int output_w = y_dims[3];
    int output_h = y_dims[2];
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, output_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, output_w);
    CL_CHECK_FATAL(status);
  } else if (kernel_func_name_ == "elementwise_add_n1h1w1") {
    int output_w = out->dims()[3];
    int output_h = out->dims()[2];
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, output_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, output_w);
    CL_CHECK_FATAL(status);
  } else if (kernel_func_name_ == "channel_add") {
    if (y->persistable()) {
      y_img = GET_DATA_GPU(y_weights_image_);
    }
    const int opt = y_dims[0] == 1;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, tensor_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, opt);
    CL_CHECK_FATAL(status);
  } else if (kernel_func_name_ == "width_add") {
    if (y->persistable()) {
      y_img = GET_DATA_GPU(y_weights_image_);
    }
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, tensor_w);
    CL_CHECK_FATAL(status);
  } else {
    LOG(FATAL) << "Unsupported kernel: " << kernel_func_name_;
  }

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

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;

// TODO(ysh329): May need fix.
// "Y" may from constant value like conv bias (kARM, need do cl_image_converter
// on CPU);
//     may from anther branch like "X" (kOpenCL, nothing to do).
// Consider 2 situations have different actions when pass running(pick kernel),
//     set target of "Y" as kOpenCL temporarily.
// REGISTER_LITE_KERNEL(elementwise_add,
//                      kOpenCL,
//                      kFP16,
//                      kImageDefault,
//                      ocl::ElementwiseAddImageCompute,
//                      def)
//     .BindInput("X",
//                {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kImageDefault))})
//     .BindInput("Y",
//                {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kImageDefault))})
//     .BindOutput("Out",
//                 {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                        PRECISION(kFP16),
//                                        DATALAYOUT(kImageDefault))})
//     .Finalize();

// REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
//                      kOpenCL,
//                      kFP16,
//                      kImageDefault,
//                      ocl::ElementwiseAddImageCompute,
//                      def)
//     .BindInput("X",
//                {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kImageDefault))})
//     .BindInput("Y",
//                {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kImageDefault))})
//     .BindOutput("Out",
//                 {LiteType::GetTensorTy(TARGET(kOpenCL),
//                                        PRECISION(kFP16),
//                                        DATALAYOUT(kImageDefault))})
//     .Finalize();
// #define LITE_WITH_LOG
