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

class ReduceComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ReduceParam;

  std::string doc() const override { return "Max using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    reduce_param_ = param_.get_mutable<param_t>();
    auto& x_dims = reduce_param_->X->dims();
    auto& dim = reduce_param_->dim;

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
          kernel_func_name_ = "reduce_n";
          break;
        case 1:
          kernel_func_name_ = "reduce_c";
          break;
        case 2:
          kernel_func_name_ = "reduce_h";
          break;
        case 3:
          kernel_func_name_ = "reduce_w";
          break;
        default:
          LOG(FATAL) << "invalid dim: " << dim[0];
      }
    } else {
      kernel_func_name_ = "max_multi_axis";
    }

    create_build_options();

    // op_type
    auto reduce_type = op_type();
    if (reduce_type == "reduce_max" || reduce_type == "max") {
      build_options_ += " -DOPERATOR(a,b)=fmax(a,b) ";
    } else if (reduce_type == "reduce_min" || reduce_type == "min") {
      build_options_ += " -DOPERATOR(a,b)=fmin(a,b) ";
    } else if (reduce_type == "reduce_sum") {
      build_options_ += " -DOPERATOR(a,b)=a+b ";
    }

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/reduce_buffer.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    std::cout << "ReInitWhenNeeded" << std::endl;
    reduce_param_ = param_.get_mutable<param_t>();
    auto& x_dims = reduce_param_->X->dims();

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
      int c = out_nchw_[1];
      int w = out_nchw_[3];
      gws_ = cl::NDRange{static_cast<cl::size_type>(hb),
                         static_cast<cl::size_type>(c),
                         static_cast<cl::size_type>(w)};
    }
  }

  void Run() override {
    std::cout << "Run" << std::endl;
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x_buf = GET_BUFFER_GPU(reduce_param_->X);
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? reduce_param_->Out->mutable_data<half_t, cl::Buffer>(
                  TARGET(kOpenCL))
            : reduce_param_->Out->mutable_data<float, cl::Buffer>(
                  TARGET(kOpenCL));
    // const auto* x_img = GET_DATA_GPU(reduce_param_->X);
    // auto out_image_shape = InitImageDimInfoWith(DDim(out_nchw_));
    // auto* out_img = MUTABLE_DATA_GPU(reduce_param_->Out,
    //                                  out_image_shape["width"],
    //                                  out_image_shape["height"],
    //                                  nullptr);
    int c4_n = in_nchw_[1] / 4;
    int c4_r = in_nchw_[1] % 4;
    int cw4 = in_nchw_[3] * c4_n;
    std::cout << "Run 1" << std::endl;
    int axis_n = 0;
    int axis_nhwc[] = {0, 0, 0, 0};
    auto dimsize = reduce_param_->dim.size();

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
      for (auto i = 0; i < reduce_param_->dim.size(); i++) {
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
    std::cout << "Run 2" << std::endl;
    int in_dims[] = {static_cast<int>(in_nchw_[0]),
                     static_cast<int>(in_nchw_[1]),
                     static_cast<int>(in_nchw_[2]),
                     static_cast<int>(in_nchw_[3])};

    cl_int status;
    int arg_idx = 0;
    status = kernel_.setArg(arg_idx++, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, in_dims[0]);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, in_dims[1]);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, in_dims[2]);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, in_dims[3]);
    CL_CHECK_FATAL(status);
    if (dimsize != 1) {
      status = kernel_.setArg(arg_idx++, axis_nhwc);
      CL_CHECK_FATAL(status);
    }
    std::cout << "Run 3" << std::endl;
    status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);
    std::cout << "Run 4" << std::endl;
  }

  void create_build_options() {
    std::string init_max = " -DDATAINIT=-FLT_MAX ";
    build_options_ = init_max;
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
  param_t* reduce_param_{nullptr};
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
                     kNCHW,
                     paddle::lite::kernels::opencl::ReduceComputeBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_sum,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::ReduceComputeBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(max,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::ReduceComputeBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
