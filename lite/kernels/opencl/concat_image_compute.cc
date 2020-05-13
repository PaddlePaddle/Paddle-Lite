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

class ConcatComputeImage : public KernelLite<TARGET(kOpenCL),
                                             PRECISION(kFP16),
                                             DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConcatParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    concat_param_ = param_.get_mutable<param_t>();
    if (concat_param_->x.size() == 2) {
      kernel_func_name_ = "concat2";
    } else {
      kernel_func_name_ = "concat_mul";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/concat_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    auto axis = concat_param_->axis;
    auto inputs = concat_param_->x;
    auto out_dims = concat_param_->output->dims();
    auto* axis_tensor = concat_param_->axis_tensor;
    if (axis_tensor != nullptr) {
      // auto* axis_tensor_data = axis_tensor->data<int>(TARGET(kARM));
      // axis = axis_tensor_data[0];
    }
    auto in_dims = inputs[0]->dims();
    axis_size_ = out_dims[axis];
    axis_ = axis;
    if (out_dims.size() < 4) {
      if (out_dims.size() - axis == 1) {
        // width
        width_ = out_dims[1];  // c
        flag_ = 3;
      } else {
        // height
        width_ = out_dims[0];  // n
        flag_ = 2;
      }
    } else {
      switch (axis_) {
        case 0:
          width_ = out_dims[2];  // h
          flag_ = 0;
          break;
        case 1:                  // channel
          width_ = out_dims[3];  // w
          flag_ = 1;
          break;
        case 2:                  // height
          width_ = out_dims[0];  // n
          flag_ = 2;
          break;
        case 3:
        case -1:                 // width
          width_ = out_dims[1];  // c
          flag_ = 3;
          break;
        default:
          printf("this axis: %d does not support \n", axis_);
      }
    }

    for (int i = 1; i < inputs.size(); i++) {
      auto dims = inputs[i]->dims();
      // auto flag = CHECK_EQ_OR_FALSE(in_dims.size(), dims.size());
      if (in_dims.size() != dims.size()) {
        printf("input shape must be same \n");
        return;
      }
      for (int i = 0; i < dims.size(); i++) {
        if (i != axis) {
          if (in_dims[i] != dims[i]) {
            printf("input shape must be same \n");
            return;
          }
        }
      }
    }
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.output->dims();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* out_buf = param.output->mutable_data<half_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);
    const auto& y_dims = param.output->dims();  // useless: check dim only

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;

    auto inputs = param.x;
    int arg_idx = 0;
    int width = inputs[0]->dims()[inputs[0]->dims().size() - 1];

#ifdef LITE_WITH_LOG
    VLOG(4) << "concat input shape:  ";
    for (size_t i = 0; i < inputs.size(); i++) {
      VLOG(4) << "inputs [" << i << "]"
              << "[" << inputs[i]->dims().size() << "D]:"
              << "   dims:" << inputs[i]->dims()[0] << " "
              << inputs[i]->dims()[1] << " " << inputs[i]->dims()[2] << " "
              << inputs[i]->dims()[3];
    }

    VLOG(4) << "concat output shape:  ";
    VLOG(4) << " out  dims:  "
            << "[" << x_dims.size() << "D]:" << x_dims[0] << " " << x_dims[1]
            << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "axis_: " << axis_;
    VLOG(4) << "flag_: " << flag_;
#endif

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(x_dims[x_dims.size() - 1]),
                    static_cast<cl::size_type>(image_shape["width"] /
                                               x_dims[x_dims.size() - 1]),
                    static_cast<cl::size_type>(image_shape["height"])};

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(param.output->target());
    VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3]
            << "x_dims[x_dims.size() - 1]" << x_dims[x_dims.size() - 1];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "width_: " << width_ << ", flag_: " << flag_;
    VLOG(4) << "global_work_size: " << x_dims[x_dims.size() - 1] << "  "
            << (image_shape["width"] / x_dims[x_dims.size() - 1]) << "  "
            << (image_shape["height"]);
#endif

    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    int out_w = x_dims[x_dims.size() - 1];
    int out_c = x_dims[1];
    if (inputs.size() == 2) {
      auto* x_buf0 = inputs[0]->data<half_t, cl::Image2D>();
      auto* x_buf1 = inputs[1]->data<half_t, cl::Image2D>();
      cl_int status = kernel.setArg(arg_idx, *x_buf0);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *x_buf1);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_buf);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, flag_);
      CL_CHECK_FATAL(status);
      status =
          kernel.setArg(++arg_idx, static_cast<int>(inputs[0]->dims()[axis_]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, out_c);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, out_w);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, width_);
      CL_CHECK_FATAL(status);

      status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
          kernel,
          cl::NullRange,
          global_work_size,
          cl::NullRange,
          nullptr,
          nullptr);
      CL_CHECK_FATAL(status);
    } else {
      auto start = 0;
      for (int i = 0; i < inputs.size(); i++) {
        arg_idx = 0;
        auto in_dims = inputs[i]->dims();
        image_shape = InitImageDimInfoWith(in_dims);
        auto* x_buf = inputs[i]->data<half_t, cl::Image2D>();
        int in_w = in_dims[in_dims.size() - 1];
#ifdef LITE_WITH_LOG
        VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
                << image_shape["height"];
#endif
        global_work_size =
            cl::NDRange{static_cast<cl::size_type>(in_dims[in_dims.size() - 1]),
                        static_cast<cl::size_type>(image_shape["width"] /
                                                   in_dims[in_dims.size() - 1]),
                        static_cast<cl::size_type>(image_shape["height"])};
        cl_int status = kernel.setArg(arg_idx, *x_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *out_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, flag_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, start);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, out_c);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, out_w);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, in_w);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, width_);
        CL_CHECK_FATAL(status);
        CL_CHECK_FATAL(status);

        status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            global_work_size,
            cl::NullRange,
            nullptr,
            nullptr);
        CL_CHECK_FATAL(status);
        start += inputs[i]->dims()[axis_];
      }
    }
  }

  std::string doc() { return "Concat using cl::Image, kFP16"; }

  int axis_size_ = 1;
  int axis_ = 1;
  int flag_ = 1;
  int width_ = 1;
  param_t* concat_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{" -DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::opencl::ConcatComputeImage Concat_image;

REGISTER_LITE_KERNEL(
    concat, kOpenCL, kFP16, kImageDefault, Concat_image, ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
