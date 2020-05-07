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
#include "lite/core/op_registry.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ElementwiseAddImageCompute::PrepareForRun() {}

void ElementwiseAddImageCompute::ReInitWhenNeeded() {
  ele_param_ = param_.get_mutable<param_t>();
  auto x_dims = ele_param_->X->dims();
  if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    first_epoch_for_reinit_ = false;

    // choose kernel
    auto* x = ele_param_->X;
    auto* y = ele_param_->Y;
    auto* out = ele_param_->Out;
    auto axis = ele_param_->axis;

    if (y->dims().size() == 4) {
      kernel_func_name_ = "elementwise_add";  // y: ImageDefault
    } else if (y->dims().size() == 1) {
      if (axis == x->dims().size() - 1) {
        kernel_func_name_ = "width_add";  // y: ImageDefault
      } else if (axis == x->dims().size() - 3) {
        kernel_func_name_ = "channel_add";  // y: ImageFolder
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

    // compute image shape
    paddle::lite::CLImageConverterDefault default_convertor;
    x_img_shape_ = default_convertor.InitImageDimInfoWith(x->dims());  // w, h
    y_img_shape_ = default_convertor.InitImageDimInfoWith(y->dims());
    out_img_shape_ =
        default_convertor.InitImageDimInfoWith(out->dims());  // w, h

    // compute global work size
    GetGlobalWorkSize();
  }
}

void ElementwiseAddImageCompute::GetGlobalWorkSize() {
  global_work_size_ = cl::NDRange{static_cast<cl::size_type>(x_img_shape_[0]),
                                  static_cast<cl::size_type>(x_img_shape_[1])};
#ifdef LITE_WITH_LOG
  VLOG(4) << "global_work_size:[2D]:" << x_img_shape_[0] << " "
          << x_img_shape_[1];
#endif
}

void ElementwiseAddImageCompute::Run() {
  auto* x = ele_param_->X;
  auto* y = ele_param_->Y;
  auto* out = ele_param_->Out;
  auto axis = ele_param_->axis;
  auto x_dims = x->dims();
  auto y_dims = y->dims();

  auto* x_img = x->data<half_t, cl::Image2D>();
  auto* y_img = y->data<half_t, cl::Image2D>();
  auto* out_img = out->mutable_data<half_t, cl::Image2D>(out_img_shape_[0],
                                                         out_img_shape_[1]);

#ifdef LITE_WITH_LOG
  VLOG(4) << "x->target():" << TargetToStr(x->target());
  VLOG(4) << "y->target():" << TargetToStr(y->target());
  VLOG(4) << "out->target():" << TargetToStr(out->target());
  VLOG(4) << "x->dims():" << x->dims();
  VLOG(4) << "y->dims():" << y->dims();
  VLOG(4) << "out->dims():" << out->dims();
  VLOG(4) << "axis:" << axis;

  VLOG(4) << "x_img_shape_[w,h]:" << x_img_shape_[0] << " " << x_img_shape_[1];
  VLOG(4) << "y_img_shape_[w,h]:" << y_img_shape_[0] << " " << y_img_shape_[1];
  VLOG(4) << "out_img_shape_[w,h]:" << out_img_shape_[0] << " "
          << out_img_shape_[1];
#endif

  cl_int status;
  auto kernel = kernel_;
  if (y_dims.size() == 4) {
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
  } else if (y_dims.size() == 1) {
    if (axis == x_dims.size() - 1 || axis == x_dims.size() - 3) {
      const int tensor_w = x_dims[x_dims.size() - 1];
#ifdef LITE_WITH_LOG
      VLOG(4) << "tensor_w:" << tensor_w;
#endif
      status = kernel.setArg(0, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, *out_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, tensor_w);
      CL_CHECK_FATAL(status);
    } else {
      LOG(FATAL) << "ElementwiseAddImage doesn't support axis:" << axis
                 << ", x->dims().size():" << x_dims.size()
                 << ", y->dims.size():" << y_dims.size();
    }
  } else {
    LOG(FATAL) << "ElementwiseAddImage doesn't support axis:" << axis
               << ", x->dims().size():" << x_dims.size()
               << ", y->dims.size():" << y_dims.size();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size_,
      cl::NullRange,
      nullptr,
      nullptr);
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
REGISTER_LITE_KERNEL(elementwise_add,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseAddImageCompute,
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
