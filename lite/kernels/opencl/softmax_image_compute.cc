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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class SoftmaxComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                                PRECISION(kFP16),
                                                DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::SoftmaxParam;

  std::string doc() const override {
    return "Softmax using cl::Image2D, kFP16";
  }

  void PrepareForRun() override {
    softmax_param_ = param_.get_mutable<param_t>();
    auto x_dims = softmax_param_->x->dims();
    int axis = softmax_param_->axis;
    axis = axis < 0 ? x_dims.size() + axis : axis;
    axis_ = 4 - x_dims.size() + axis;
    if (axis_ == 3) {
      kernel_func_name_ = "softmax_width";
    } else if (axis_ == 2) {
      kernel_func_name_ = "softmax_height";
    } else if (axis_ == 1) {
      kernel_func_name_ = "softmax_channel";
    } else {
      LOG(FATAL) << "do not support this axis value!"
                 << "axis value is: " << axis_;
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/softmax_kernel.cl",
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
    auto out_dims = softmax_param_->output->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      out_img_shape_ = default_convertor.InitImageDimInfoWith(out_dims);

      // compute global work size
      const std::vector<size_t>& default_work_size =
          DefaultGlobalWorkSize(out_dims,
                                DDim(std::vector<DDim::value_type>{
                                    static_cast<int64_t>(out_img_shape_[0]),
                                    static_cast<int64_t>(out_img_shape_[1])}));
      int c_blk = default_work_size.data()[0];
      int w = default_work_size.data()[1];
      int bh = default_work_size.data()[2];
      if (axis_ == 3) {
        global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c_blk),
                                        static_cast<cl::size_type>(bh),
                                        static_cast<cl::size_type>(1)};
      } else if (axis_ == 2) {
        global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c_blk * w),
                                        static_cast<cl::size_type>(out_dims[0]),
                                        static_cast<cl::size_type>(1)};
      } else {
        global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c_blk),
                                        static_cast<cl::size_type>(w),
                                        static_cast<cl::size_type>(bh)};
      }
    }
  }

  void Run() override {
    auto x_dims = softmax_param_->x->dims();
    int input_dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < x_dims.size(); i++) {
      input_dims[4 - x_dims.size() + i] = x_dims[i];
    }
    auto* x_img = GET_DATA_GPU(softmax_param_->x);
    auto* out_img = MUTABLE_DATA_GPU(
        softmax_param_->output, out_img_shape_[0], out_img_shape_[1], nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);
    if (axis_ == 3 || axis_ == 2) {
      status = kernel.setArg(2, input_dims[0]);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, input_dims[1]);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, input_dims[2]);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, input_dims[3]);
      CL_CHECK_FATAL(status);
    } else {
      int c_blk = global_work_size_[0];
      int w = global_work_size_[1];
      int c_remain = c_blk * 4 - input_dims[1];
      status = kernel.setArg(2, input_dims[1]);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, c_remain);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, c_blk);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, w);
      CL_CHECK_FATAL(status);
    }

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
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"softmax_width"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* softmax_param_{nullptr};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  int axis_{3};
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(softmax,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SoftmaxComputeImage2D,
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
