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

#include <vector>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class MaxComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                            PRECISION(kFP16),
                                            DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ReduceParam;

  std::string doc() const override { return "Max using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    max_param_ = param_.get_mutable<param_t>();
    auto& x_dims = max_param_->X->dims();
    auto& dim = max_param_->dim;

    // padding to 4-dims
    in_nchw_ = x_dims.Vectorize();
    while (in_nchw_.size() < 4) {
      in_nchw_.insert(in_nchw_.cbegin(), 1);
    }

    // format axis
    int offset = 4 - x_dims.size();
    for (auto i = 0; i < dim.size(); i++) {
      axis_.push_back(dim[i] >= 0 ? dim[i] + offset
                                  : dim[i] + x_dims.size() + offset);
    }

    if (dim.size() == 1) {
      switch (axis_[0]) {
        case 0:
          kernel_func_name_ = "max_n";
          break;
        case 1:
          kernel_func_name_ = "max_c";
          break;
        case 2:
          kernel_func_name_ = "max_h";
          break;
        case 3:
          kernel_func_name_ = "max_w";
          break;
        default:
          LOG(FATAL) << "invalid dim: " << dim[0];
      }
    } else {
      kernel_func_name_ = "max_multi_axis";
    }

    create_build_options();

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/max_kernel.cl", build_options_, time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    max_param_ = param_.get_mutable<param_t>();
    auto& x_dims = max_param_->X->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute global work size
      // padding out_dims to 4-dims
      out_nchw_ = in_nchw_;
      for (auto k = 0; k < axis_.size(); k++) {
        out_nchw_[axis_[k]] = 1;
      }

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

    const auto* x_img = GET_DATA_GPU(max_param_->X);
    auto out_image_shape = InitImageDimInfoWith(DDim(out_nchw_));
    auto* out_img = MUTABLE_DATA_GPU(max_param_->Out,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);
    int c4_n = in_nchw_[1] / 4;
    int c4_r = in_nchw_[1] % 4;
    int cw4 = in_nchw_[3] * c4_n;

    int axis_n = 0;
    int axis_nhwc[] = {0, 0, 0, 0};
    auto dimsize = max_param_->dim.size();

    if (dimsize == 0) {
      axis_n = std::accumulate(
          in_nchw_.cbegin(), in_nchw_.cend(), 0, std::multiplies<int64_t>());
      axis_nhwc[0] = 1;
      axis_nhwc[1] = 1;
      axis_nhwc[2] = 1;
      axis_nhwc[3] = 1;
    } else if (dimsize == 1) {
      axis_n = in_nchw_[axis_[0]];
    } else {
      // multi axies
      axis_n = 1;
      for (auto i = 0; i < max_param_->dim.size(); i++) {
        int axis = axis_[i];
        switch (axis) {
          case 0:  // n
            if (!axis_nhwc[0]) {
              axis_n *= in_nchw_[axis];
              axis_nhwc[0] = 1;
            }
            break;
          case 1:  // c
            if (!axis_nhwc[3]) {
              axis_n *= in_nchw_[axis];
              axis_nhwc[3] = 1;
            }
            break;
          case 2:  // h
            if (!axis_nhwc[1]) {
              axis_n *= in_nchw_[axis];
              axis_nhwc[1] = 1;
            }
            break;
          case 3:  // w
            if (!axis_nhwc[2]) {
              axis_n *= in_nchw_[axis];
              axis_nhwc[2] = 1;
            }
            break;
          default:
            LOG(FATAL) << "invalid axis: " << axis;
        }
      }
    }

    int in_dims[] = {static_cast<int>(in_nchw_[0]),
                     static_cast<int>(in_nchw_[1]),
                     static_cast<int>(in_nchw_[2]),
                     static_cast<int>(in_nchw_[3])};

    cl_int status;
    int arg_idx = 0;
    status = kernel_.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, in_dims);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, c4_n);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, c4_r);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, cw4);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, axis_n);
    CL_CHECK_FATAL(status);
    if (dimsize != 1) {
      status = kernel_.setArg(arg_idx++, axis_nhwc);
      CL_CHECK_FATAL(status);
    }

    status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);
  }

  void create_build_options() {
    std::string init_fp32 = " -DDATAINIT=-FLT_MAX ";
    std::string init_fp16 = " -DDATAINIT=-HALF_MAX ";
    build_options_ =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? init_fp16
            : init_fp32;
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->global_work_size = ch->NDRangeToStr(gws_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  param_t* max_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::vector<int64_t> in_nchw_{};
  std::vector<int64_t> out_nchw_{};
  std::vector<int> axis_{};
  std::string kernel_func_name_{};
  std::string build_options_{};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_max,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::MaxComputeImage2D,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(max,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::MaxComputeImage2D,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
