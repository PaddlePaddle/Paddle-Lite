// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

static void HelperFunc(const DDimLite& in_dims,
                       const int axis,
                       std::string* kernel_func_name,
                       int* height) {
  if (in_dims.size() < 4) {
    if (in_dims.size() == 3) {
      switch (axis) {
        case 0:
          *kernel_func_name = "SplitChannel";
          break;
        case 1:
          *kernel_func_name = "SplitHeight";
          break;
        case 2:
          *kernel_func_name = "SplitWidth";
          break;
        default:
          LOG(FATAL) << "Unsupported axis: " << axis << " for this dimension";
      }
      *height = in_dims[1];
    } else if (in_dims.size() == 2) {
      switch (axis) {
        case 0:
          *kernel_func_name = "SplitHeight";
          break;
        case 1:
          *kernel_func_name = "SplitWidth";
          break;
        default:
          LOG(FATAL) << "Unsupported axis: " << axis << " for this dimension";
      }
      *height = in_dims[0];
    } else {
      switch (axis) {
        case 0:
          *kernel_func_name = "SplitWidth";
          break;
        default:
          LOG(FATAL) << "Unsupported axis: " << axis << " for this dimension";
      }
    }
  } else {
    switch (axis) {
      case 0:
        *kernel_func_name = "SplitBatch";
        break;
      case 1:
        *kernel_func_name = "SplitChannel";
        break;
      case 2:
        *kernel_func_name = "SplitHeight";
        break;
      case 3:
        *kernel_func_name = "SplitWidth";
        break;
      default:
        LOG(FATAL) << "Unsupported axis: " << axis << " for this dimension";
    }
    *height = in_dims[2];
  }
}

class SplitComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::SplitParam;

  std::string doc() const override { return "Split using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    split_param_ = param_.get_mutable<param_t>();
    auto& x_dims = split_param_->x->dims();
    auto& outs = split_param_->output;
    axis_ = split_param_->axis;
    if (axis_ < 0) {
      axis_ += x_dims.size();
    }
    VLOG(4) << "axis: " << axis_;
    VLOG(4) << "x_dims: " << x_dims;
    VLOG(4) << "outs.size(): " << outs.size();
    VLOG(4) << "num: " << split_param_->num;
    if (outs.size() != 2) {
      LOG(FATAL) << "NOT imple yet!";
    }
    HelperFunc(x_dims, axis_, &kernel_func_name_, &height_);

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/split_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    split_param_ = param_.get_mutable<param_t>();
    auto& x_dims = split_param_->x->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute global work size
      auto x_img_shape = InitImageDimInfoWith(x_dims);
      const auto& default_work_size = DefaultGlobalWorkSize(
          x_dims,
          DDim(std::vector<DDim::value_type>{
              static_cast<int64_t>(x_img_shape["width"]),
              static_cast<int64_t>(x_img_shape["height"])}));
      gws_ = cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                         static_cast<cl::size_type>(default_work_size[1]),
                         static_cast<cl::size_type>(default_work_size[2])};
      VLOG(4) << "gws: " << gws_[0] << ", " << gws_[1] << ", " << gws_[2];
    }
  }

  void Run() override {
    const auto& x_dims = split_param_->x->dims();
    const int out0_dims_axis = split_param_->output[0]->dims()[axis_];
    const auto out_num = split_param_->output.size();
    const auto* x_img = GET_DATA_GPU(split_param_->x);

    std::vector<cl::Image2D*> out_img{out_num};
    for (auto i = 0; i < split_param_->output.size(); i++) {
      auto image_shape = InitImageDimInfoWith(split_param_->output[i]->dims());
      out_img[i] = MUTABLE_DATA_GPU(split_param_->output[i],
                                    image_shape["width"],
                                    image_shape["height"],
                                    nullptr);
      VLOG(4) << "out_dims: " << split_param_->output[i]->dims();
      VLOG(4) << "out_img[w, h]: " << image_shape["width"] << ", "
              << image_shape["height"];
    }
    VLOG(4) << "out0_dims_axis: " << out0_dims_axis;
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    if (out_num == 2) {
      cl_int status;
      int arg_idx = 0;
      status = kernel_.setArg(arg_idx++, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(arg_idx++, *(out_img[0]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(arg_idx++, *(out_img[1]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(arg_idx++, out0_dims_axis);
      CL_CHECK_FATAL(status);
      if (kernel_func_name_ == "SplitBatch" ||
          kernel_func_name_ == "SplitHeight") {
        status = kernel_.setArg(arg_idx++, height_);
        CL_CHECK_FATAL(status);
      }
      status = kernel_.setArg(arg_idx++,
                              static_cast<int>(x_dims[x_dims.size() - 1]));
      CL_CHECK_FATAL(status);

      status = EnqueueNDRangeKernel(context,
                                    kernel_,
                                    cl::NullRange,
                                    gws_,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else {
      LOG(FATAL) << "NOT imple yet!";
    }
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  int axis_{-1};
  int height_{-1};
  param_t* split_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(split,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SplitComputeImage2D,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
