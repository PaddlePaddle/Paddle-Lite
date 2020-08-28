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

#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ConcatCompute : public KernelLite<TARGET(kOpenCL),
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
      kernel_func_name_ = "concat_mul_buffer";
    }
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/concat_kernel.cl",
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
    for (int i = 0; i < axis; i++) {
      pre_size_ *= in_dims[i];
    }
    for (int i = axis + 1; i < in_dims.size(); i++) {
      post_size_ *= in_dims[i];
    }

    for (int i = 1; i < inputs.size(); i++) {
      auto dims = inputs[i]->dims();
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
    auto* out_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    const auto& y_dims = param.output->dims();  // useless: check dim only

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;

    auto inputs = param.x;
    int arg_idx = 0;
    auto global_work_size = cl::NDRange{static_cast<cl::size_type>(axis_size_)};
    int total = axis_size_ * post_size_;

    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    if (inputs.size() == 2) {
      auto* x_buf0 = inputs[0]->data<float, cl::Buffer>();
      auto* x_buf1 = inputs[1]->data<float, cl::Buffer>();
      auto axis0 = inputs[0]->dims()[axis_];
      int total0 = axis0 * post_size_;
      int total1 = (axis_size_ - axis0) * post_size_;
      cl_int status = kernel.setArg(arg_idx, *x_buf0);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *x_buf1);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_buf);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<int>(axis0));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, axis_size_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, pre_size_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, post_size_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, total);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, total0);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, total1);
      CL_CHECK_FATAL(status);

      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else {
      auto start = 0;
      for (int i = 0; i < inputs.size(); i++) {
        arg_idx = 0;
        int size = inputs[i]->dims()[axis_];
        auto* x_buf = inputs[i]->data<float, cl::Buffer>();
        global_work_size = cl::NDRange{static_cast<size_t>(size)};
        int total0 = size * post_size_;
#ifdef LITE_WITH_LOG
        LOG(INFO) << "------------- i=" << i << " -------------";
        LOG(INFO) << "pre_size:" << pre_size_;
        LOG(INFO) << "post_size:" << post_size_;
        LOG(INFO) << "size:" << size;
        LOG(INFO) << "start:" << start;
        LOG(INFO) << "total:" << total;
        LOG(INFO) << "total0:" << total0;
#endif
        cl_int status = kernel.setArg(arg_idx, *x_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *out_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, static_cast<int>(size));
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, pre_size_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, post_size_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, start);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, total);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, total0);
        CL_CHECK_FATAL(status);

        status = EnqueueNDRangeKernel(context,
                                      kernel,
                                      cl::NullRange,
                                      global_work_size,
                                      cl::NullRange,
                                      nullptr,
                                      event_);
        CL_CHECK_FATAL(status);
        start += size;
      }
    }
  }

  std::string doc() { return "Concat using cl::Buffer, kFloat"; }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  int axis_size_ = 1;
  int post_size_ = 1;
  int pre_size_ = 1;
  int axis_ = 1;
  param_t* concat_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::opencl::ConcatCompute Concat_buffer;

REGISTER_LITE_KERNEL(concat, kOpenCL, kFloat, kNCHW, Concat_buffer, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
