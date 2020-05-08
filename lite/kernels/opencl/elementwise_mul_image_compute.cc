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
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/logging.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ElementwiseMulImageCompute
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ElementwiseParam;

  std::string doc() const override {
    return "ElementwiseMul using cl::Image2D(ImageDefault/RGBA), kFP32";
  }

  void PrepareForRun() override {
    ele_param_ = param_.get_mutable<param_t>();
    auto* y = ele_param_->Y;
    auto* x = ele_param_->X;
    auto bias_dims = y->dims();
    auto x_dims = x->dims();

    if (bias_dims == x_dims) {
      kernel_func_name_ = "elementwise_mul";
    } else {
      const int bias_dim_size = bias_dims.size();
      if (bias_dim_size == 1) {
        kernel_func_name_ = "channel_mul_d1";
      } else if (bias_dim_size == 2) {
        kernel_func_name_ = "channel_mul_d2";
      } else if (bias_dim_size == 3) {
        kernel_func_name_ = "channel_mul_d3";
      } else if (bias_dim_size == 4) {
        kernel_func_name_ = "channel_mul_d4";
      } else {
        LOG(FATAL) << "Unsupported ElementwiseMul with x_dims:" << x_dims
                   << " y_dims:" << bias_dims;
      }
    }

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    VLOG(4) << "x_dims:" << x_dims;
    VLOG(4) << "bias_dims:" << bias_dims;
    VLOG(4) << "bias_dims.size():" << bias_dims.size();

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/elementwise_mul_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x = ele_param_->X;
    auto* y = ele_param_->Y;
    auto* out = ele_param_->Out;

#ifdef LITE_WITH_LOG
    VLOG(4) << "x->target():" << TargetToStr(x->target());
    VLOG(4) << "y->target():" << TargetToStr(y->target());
    VLOG(4) << "out->target():" << TargetToStr(out->target());
    VLOG(4) << "x->dims():" << x->dims();
    VLOG(4) << "y->dims():" << y->dims();
    VLOG(4) << "out->dims():" << out->dims();
#endif

    paddle::lite::CLImageConverterDefault default_convertor;
    auto x_img_shape =
        default_convertor.InitImageDimInfoWith(x->dims());  // w, h
    auto x_img_width = x_img_shape[0];
    auto x_img_height = x_img_shape[1];
    auto out_img_shape =
        default_convertor.InitImageDimInfoWith(out->dims());  // w, h
    auto y_img_shape = default_convertor.InitImageDimInfoWith(y->dims());

    auto* x_img = x->data<half_t, cl::Image2D>();
    auto* y_img = y->data<half_t, cl::Image2D>();
    auto* out_img = out->mutable_data<half_t, cl::Image2D>(out_img_shape[0],
                                                           out_img_shape[1]);

#ifdef LITE_WITH_LOG
    VLOG(4) << "x_img_shape[w,h]:" << x_img_width << " " << x_img_height;
    VLOG(4) << "y_img_shape[w,h]:" << y_img_shape[0] << " " << y_img_shape[1];
    VLOG(4) << "out_img_shape[w,h]:" << out_img_shape[0] << " "
            << out_img_shape[1];
#endif

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    auto bias_dims = y->dims();
    auto x_dims = x->dims();

    if (bias_dims == x_dims) {
      // kernel_func_name_ = "elementwise_mul";
      cl_int status = kernel.setArg(0, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, *out_img);
      CL_CHECK_FATAL(status);
    } else {
      const int bias_dim_size = bias_dims.size();
      if (bias_dim_size == 1) {
        // kernel_func_name_ = "channel_mul_d1";
        const int tensor_w = x_dims[x_dims.size() - 1];
        cl_int status = kernel.setArg(0, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, tensor_w);
        CL_CHECK_FATAL(status);
      } else if (bias_dim_size == 2) {
        // kernel_func_name_ = "channel_mul_d2";
        const int tensor_w = x_dims[x_dims.size() - 1];
        cl_int status = kernel.setArg(0, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, tensor_w);
        CL_CHECK_FATAL(status);
      } else if (bias_dim_size == 3) {
        // kernel_func_name_ = "channel_mul_d3";
        const int tensor_w = x_dims[x_dims.size() - 1];
        cl_int status = kernel.setArg(0, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, tensor_w);
        CL_CHECK_FATAL(status);
      } else if (bias_dim_size == 4) {
        // kernel_func_name_ = "channel_mul_d4";
        const int tensor_w = x_dims[x_dims.size() - 1];
        cl_int status = kernel.setArg(0, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, tensor_w);
        CL_CHECK_FATAL(status);
      } else {
        LOG(FATAL) << "Unsupported ElementwiseMul with x_dims:" << x_dims
                   << " y_dims:" << bias_dims;
      }
    }

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(x_img_width),
                    static_cast<cl::size_type>(x_img_height)};

    auto status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        nullptr);
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:[2D]:" << x_img_width << " " << x_img_height;
#endif
  }

 protected:
  param_t* ele_param_{nullptr};
  std::string kernel_func_name_{"elementwise_mul"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(elementwise_mul,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseMulImageCompute,
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
