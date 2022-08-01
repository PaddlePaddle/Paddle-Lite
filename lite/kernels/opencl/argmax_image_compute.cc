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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ArgmaxComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ArgmaxParam;

  std::string doc() const override { return "Argmax using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    argmax_param_ = param_.get_mutable<param_t>();
    auto& x_dims = argmax_param_->X->dims();
    bool keepdims = argmax_param_->keepdims;
    CHECK(keepdims) << "OpenCL argmax kernel only support keepdims=true. "
                       "keepdims=false case will be converted by "
                       "keepdims_convert_pass.";

    // padding to 4-dims
    in_nchw_ = x_dims.Vectorize();
    while (in_nchw_.size() < 4) {
      in_nchw_.insert(in_nchw_.cbegin(), 1);
    }

    axis_ = argmax_param_->Axis;
    if (axis_ < 0) axis_ += x_dims.size();
    int padding_axis = axis_ + (4 - x_dims.size());
    switch (padding_axis) {
      case 0:
        kernel_func_name_ = "argmax_n";
        break;
      case 1:
        kernel_func_name_ = "argmax_c";
        break;
      case 2:
        kernel_func_name_ = "argmax_h";
        break;
      case 3:
        kernel_func_name_ = "argmax_w";
        break;
      default:
        LOG(FATAL) << "invalid axis: " << argmax_param_->Axis;
    }

    create_build_options();

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/argmax_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    argmax_param_ = param_.get_mutable<param_t>();
    auto& x_dims = argmax_param_->X->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute global work size
      // padding out_dims to 4-dims
      out_nchw_ = in_nchw_;
      out_nchw_[axis_ + (4 - x_dims.size())] = 1;

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

    const auto* x_img = GET_DATA_GPU(argmax_param_->X);
    auto out_image_shape = InitImageDimInfoWith(DDim(out_nchw_));
    auto* out_img = MUTABLE_DATA_GPU(argmax_param_->Out,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);
    int c4_n = in_nchw_[1] / 4;
    int c4_r = in_nchw_[1] % 4;
    int cw4 = in_nchw_[3] * c4_n;

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

    status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);
  }

  void create_build_options() {
    const bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    std::string init_max = " -DDATAINIT=-FLT_MAX ";
    std::string flag_type =
        fp16_support ? " -DFLAG_TYPE4=short4 " : " -DFLAG_TYPE4=int4 ";

    build_options_ = init_max + flag_type;
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
  param_t* argmax_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  int axis_{-1};
  std::vector<int64_t> in_nchw_{};
  std::vector<int64_t> out_nchw_{};
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

REGISTER_LITE_KERNEL(arg_max,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ArgmaxComputeImage2D,
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
