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

  std::string doc() const override { return "Softmax using cl::Buffer, kFP16"; }

  void PrepareForRun() override {
    softmax_param_ = param_.get_mutable<param_t>();
    auto x_dims = softmax_param_->x->dims();
    int axis = softmax_param_->axis;
    VLOG(4) << "x_dims: " << x_dims;
    VLOG(4) << "axis: " << axis;
    auto extend_in_dims = ExtendInputDims(x_dims);
    axis = axis < 0 ? x_dims.size() + axis : axis;
    axis_ = 4 - x_dims.size() + axis;

    if (extend_in_dims[3] < 4) {
      small_w_flag_ = true;
      kernel_func_name_ = "softmax_common_buffer";
    } else if ((x_dims.size() == 2 || x_dims.size() == 1) && axis_ == 3) {
      onexone_flag_ = true;
      kernel_func_name_ = "softmax_1x1_buffer";
    } else if (axis_ == 3) {
      kernel_func_name_ = "softmax_width_buffer";
    } else if (axis_ == 2) {
      kernel_func_name_ = "softmax_height_buffer";
    } else if (axis_ == 1) {
      kernel_func_name_ = "softmax_channel_buffer";
    } else if (axis_ == 0) {
      kernel_func_name_ = "softmax_batch_buffer";
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

    if (small_w_flag_) {
      int select_dim = 1;
      for (int i = extend_in_dims.size() - 1; i >= 0; i--) {
        if (i > axis_) {
          select_dim *= extend_in_dims[i];
        }
      }
      int pre_dim = extend_in_dims[axis_] * select_dim;
      status = kernel.setArg(2, static_cast<int>(pre_dim));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, static_cast<int>(extend_in_dims[axis_]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, static_cast<int>(select_dim));
      CL_CHECK_FATAL(status);
    } else if (onexone_flag_) {
      status = kernel.setArg(2, static_cast<int>(extend_in_dims[3]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, UP_DIV(static_cast<int>(extend_in_dims[3]), 4));
      CL_CHECK_FATAL(status);
    } else {
      status = kernel.setArg(2, static_cast<int>(extend_in_dims[0]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, static_cast<int>(extend_in_dims[1]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, static_cast<int>(extend_in_dims[2]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, static_cast<int>(extend_in_dims[3]));
      CL_CHECK_FATAL(status);
      if (axis_ == 3) {
        auto mask_v = GetMask4(extend_in_dims[3]);
        cl_float4 mask = {mask_v[0], mask_v[1], mask_v[2], mask_v[3]};
        status = kernel.setArg(6, mask);
        CL_CHECK_FATAL(status);
      }
    }

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  local_work_size_,
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
    auto extend_in_dims = ExtendInputDims(x_dims);
    int n = extend_in_dims[0];
    int c = extend_in_dims[1];
    int h = extend_in_dims[2];
    int w = extend_in_dims[3];
    int w_blk = (w + 3) / 4;
    int bh = n * h;

    if (small_w_flag_) {
      int suffix_num = 1;
      int prefix_num = 1;
      for (int i = 0; i < extend_in_dims.size(); i++) {
        if (i < axis_) {
          prefix_num *= extend_in_dims[i];
        } else if (i > axis_) {
          suffix_num *= extend_in_dims[i];
        }
      }
      global_work_size_ = cl::NDRange(prefix_num, suffix_num, 1);
    } else if (onexone_flag_) {
      local_work_size_ = cl::NDRange(32, 1, 1);
      global_work_size_ =
          cl::NDRange(ROUND_UP(UP_DIV(w, 4), local_work_size_[0]), h, 1);
    } else if (axis_ == 3) {  // for width
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c),
                                      static_cast<cl::size_type>(bh),
                                      static_cast<cl::size_type>(1)};
    } else if (axis_ == 2) {  // for height
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c * w_blk),
                                      static_cast<cl::size_type>(n),
                                      static_cast<cl::size_type>(1)};
    } else if (axis_ == 1) {  // for channel
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(h * w_blk),
                                      static_cast<cl::size_type>(n),
                                      static_cast<cl::size_type>(1)};
    } else {  // for batch
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(h * w_blk),
                                      static_cast<cl::size_type>(c),
                                      static_cast<cl::size_type>(1)};
    }
    VLOG(4) << "gws: " << global_work_size_[0] << ", " << global_work_size_[1]
            << ", " << global_work_size_[2];
  }

  const std::vector<float> GetMask4(int total_count) {
    std::vector<float> mask{0.0f, 0.0f, 0.0f, 0.0f};
    const int reminder = total_count % 4 == 0 ? 4 : total_count % 4;
    for (int i = 0; i < reminder; ++i) {
      mask[3 - i] = 1.0f;
    }
    return mask;
  }

  const DDim ExtendInputDims(const DDim& in_dims) {
    auto extend_dims = std::vector<int64_t>{1, 1, 1, 1};
    for (int i = 0; i < in_dims.size(); i++) {
      extend_dims[4 - in_dims.size() + i] = in_dims[i];
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
  bool small_w_flag_{false};
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
