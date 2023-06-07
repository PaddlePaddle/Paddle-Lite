// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

class SoftmaxComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SoftmaxParam;

  std::string doc() const override {
    return "Softmax using cl::Image2D, kFP16";
  }

  void PrepareForRun() override {
    softmax_param_ = param_.get_mutable<param_t>();
    auto x_dims = softmax_param_->x->dims();
    int axis = softmax_param_->axis;
    VLOG(4) << "x_dims: " << x_dims;
    VLOG(4) << "axis: " << axis;
    if (x_dims.size() > 1) {
      axis = axis < 0 ? x_dims.size() + axis : axis;
      axis_ = 4 - x_dims.size() + axis;
    } else {      // for dim 1
      axis_ = 1;  // process width as channel for folder format
    }
    VLOG(4) << "axis_: " << axis_;
    if (x_dims.size() == 2 && axis_ == 3) {
      onexone_flag_ = true;
      kernel_func_name_ = "softmax_1x1";
    } else if (axis_ == 3) {
      kernel_func_name_ = "softmax_width_buffer";
    } else if (axis_ == 2) {
      kernel_func_name_ = "softmax_height_buffer";
    } else if (axis_ == 1) {
      kernel_func_name_ = "softmax_channel";
    } else if (axis_ == 0) {
      kernel_func_name_ = "softmax_batch";
    } else {
      LOG(FATAL) << "do not support this axis value!"
                 << "axis value is: " << axis_;
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/softmax_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    softmax_param_ = param_.get_mutable<param_t>();
    auto x_dims = softmax_param_->x->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      SetGlobalLocal();
    }
  }
  void Run() override {
    auto x_dims = softmax_param_->x->dims();
    auto extend_in_dims = ExtendInputDims(x_dims);
    auto* x_buf = GET_BUFFER_GPU(softmax_param_->x);
    auto* out_buf = MUTABLE_BUFFER_GPU(softmax_param_->output);
    VLOG(4) << "extend_in_dims: " << extend_in_dims;
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto& kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, static_cast<int>(extend_in_dims[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, static_cast<int>(extend_in_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<int>(extend_in_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<int>(extend_in_dims[3]));
    CL_CHECK_FATAL(status);

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
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void SetGlobalLocal() {
    auto x_dims = softmax_param_->x->dims();
    int c = x_dims[1];
    int w_blk = (x_dims[3] + 3) / 4;
    // int w = x_dims[3];
    int bh = x_dims[0] * x_dims[2];
    if (axis_ == 3) {  // for width
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c),
                                      static_cast<cl::size_type>(bh),
                                      static_cast<cl::size_type>(1)};
    } else if (axis_ == 2) {  // for height
      global_work_size_ =
          cl::NDRange{static_cast<cl::size_type>(c * w_blk),
                      static_cast<cl::size_type>(last_x_dims_[0]),
                      static_cast<cl::size_type>(1)};
    }
    VLOG(4) << "gws: " << global_work_size_[0] << ", " << global_work_size_[1]
            << ", " << global_work_size_[2];
  }

  const std::vector<float> GetChannelMask(int channels) {
    std::vector<float> mask{0.0f, 0.0f, 0.0f, 0.0f};
    const int reminder = channels % 4 == 0 ? 4 : channels % 4;
    for (int i = 0; i < reminder; ++i) {
      mask[i] = 1.0f;
    }
    return mask;
  }

  const DDim ExtendInputDims(const DDim& in_dims) {
    auto extend_dims = std::vector<int64_t>{1, 1, 1, 1};
    if (onexone_flag_) {
      extend_dims[0] = in_dims[0];
      extend_dims[1] = in_dims[1];
    } else {
      for (int i = 0; i < in_dims.size(); i++) {
        extend_dims[4 - in_dims.size() + i] = in_dims[i];
      }
      if (in_dims.size() ==
          1) {  // transform dim_w to dim_c for dim1 folder case
        extend_dims[1] = in_dims[0];
        extend_dims[3] = 1;
      }
    }
    return DDim(extend_dims);
  }

 private:
  std::string kernel_func_name_{""};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* softmax_param_{nullptr};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  int axis_;
  bool onexone_flag_{false};
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_{cl::NullRange};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(softmax,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::SoftmaxComputeBuffer,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
