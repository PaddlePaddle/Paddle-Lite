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
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

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

    auto inputs = concat_param_->x;
    auto axis_ = concat_param_->axis;
    auto output_tensor_dims = concat_param_->output->dims();
    auto* axis_tensor = concat_param_->axis_tensor;
    if (axis_tensor != nullptr) {
      // auto* axis_tensor_data = axis_tensor->data<int>(TARGET(kARM));
      // axis = axis_tensor_data[0];
    }

    if (inputs.size() == 2) {
      kernel_func_name_ = "concat2";
    } else if (inputs.size() == 3) {
      kernel_func_name_ = "concatByCWith3Inputs";
    } else if (inputs.size() == 4) {
      kernel_func_name_ = "concatByCWith4Inputs";
    } else {
      // TODO(ysh329): do layout transform between image and buffer,
      // before and after concat(buffer impl.)
      kernel_func_name_ = "concat_mul";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/concat_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    if (output_tensor_dims.size() < 4) {
      if (output_tensor_dims.size() - axis_ == 1) {
        // width
        width_ = output_tensor_dims[1];  // c
        flag_ = 3;
      } else {
        // height
        width_ = output_tensor_dims[0];  // n
        flag_ = 2;
      }
    } else {
      switch (axis_) {
        case 0:
          width_ = output_tensor_dims[2];  // h
          flag_ = 0;
          break;
        case 1:                            // channel
          width_ = output_tensor_dims[3];  // w
          flag_ = 1;
          break;
        case 2:                            // height
          width_ = output_tensor_dims[0];  // n
          flag_ = 2;
          break;
        case 3:
        case -1:                           // width
          width_ = output_tensor_dims[1];  // c
          flag_ = 3;
          break;
        default:
          LOG(FATAL) << "Unsupported axis:" << axis_;
      }
    }

    auto input0_tensor_dims = inputs[0]->dims();
    for (int i = 1; i < inputs.size(); i++) {
      auto dims = inputs[i]->dims();
      // auto flag = CHECK_EQ_OR_FALSE(input0_tensor_dims.size(), dims.size());
      if (input0_tensor_dims.size() != dims.size()) {
        printf("input shape must be same \n");
        return;
      }
      for (int i = 0; i < dims.size(); i++) {
        if (i != axis_) {
          if (input0_tensor_dims[i] != dims[i]) {
            printf("input shape must be same \n");
            return;
          }
        }
      }
    }
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& output_tensor_dims = param.output->dims();
    int output_tensor_w = output_tensor_dims[output_tensor_dims.size() - 1];
    int output_tensor_c = output_tensor_dims[1];
    auto output_image_shape = InitImageDimInfoWith(output_tensor_dims);
    auto* output_image_p = param.output->mutable_data<half_t, cl::Image2D>(
        output_image_shape["width"], output_image_shape["height"]);
    auto inputs = param.x;

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(
                        output_tensor_dims[output_tensor_dims.size() - 1]),
                    static_cast<cl::size_type>(
                        output_image_shape["width"] /
                        output_tensor_dims[output_tensor_dims.size() - 1]),
                    static_cast<cl::size_type>(output_image_shape["height"])};

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
            << "[" << output_tensor_dims.size()
            << "D]:" << output_tensor_dims[0] << " " << output_tensor_dims[1]
            << " " << output_tensor_dims[2] << " " << output_tensor_dims[3];
    VLOG(4) << "axis_: " << axis_;
    VLOG(4) << "flag_: " << flag_;

    VLOG(4) << TargetToStr(param.output->target());
    VLOG(4) << "output_image_shape(w,h):" << output_image_shape["width"] << " "
            << output_image_shape["height"];
    VLOG(4) << "output_tensor_dims[" << output_tensor_dims.size()
            << "D]:" << output_tensor_dims[0] << " " << output_tensor_dims[1]
            << " " << output_tensor_dims[2] << " " << output_tensor_dims[3]
            << "output_tensor_dims[output_tensor_dims.size() - 1]"
            << output_tensor_dims[output_tensor_dims.size() - 1];
    VLOG(4) << "output_tensor_w: " << output_tensor_w << ", flag_: " << flag_;
    VLOG(4) << "width_:" << width_;
    VLOG(4) << "global_work_size: "
            << output_tensor_dims[output_tensor_dims.size() - 1] << "  "
            << (output_image_shape["width"] /
                output_tensor_dims[output_tensor_dims.size() - 1])
            << "  " << (output_image_shape["height"]);
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    int arg_idx = 0;

    if (kernel_func_name_ == "concat2") {
      auto* input0_image_p = inputs[0]->data<half_t, cl::Image2D>();
      auto* input1_image_p = inputs[1]->data<half_t, cl::Image2D>();
      cl_int status = kernel.setArg(arg_idx, *input0_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *input1_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *output_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, flag_);
      CL_CHECK_FATAL(status);
      status =
          kernel.setArg(++arg_idx, static_cast<int>(inputs[0]->dims()[axis_]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, output_tensor_c);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, output_tensor_w);
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
    } else if (kernel_func_name_ == "concatByCWith3Inputs" ||
               kernel_func_name_ == "concatByCWith4Inputs") {
      auto* input0 = inputs[0];
      auto* input0_image_p = input0->data<half_t, cl::Image2D>();
      size_t input0_tensor_c = input0->dims()[1];

      auto* input1 = inputs.size() >= 2 ? inputs[1] : nullptr;
      auto* input1_image_p =
          input1 ? input1->data<half_t, cl::Image2D>() : nullptr;
      size_t input1_tensor_c = input1 ? input1->dims()[1] : -1;

      auto* input2 = inputs.size() >= 3 ? inputs[2] : nullptr;
      auto* input2_image_p =
          input2 ? input2->data<half_t, cl::Image2D>() : nullptr;
      size_t input2_tensor_c = input2 ? input2->dims()[1] : -1;

      auto* input3 = inputs.size() >= 4 ? inputs[3] : nullptr;
      auto* input3_image_p =
          input3 ? input3->data<half_t, cl::Image2D>() : nullptr;
      size_t input3_tensor_c = input3 ? input3->dims()[1] : -1;

      const std::vector<size_t>& default_work_size = DefaultWorkSize(
          output_tensor_dims,
          DDim(std::vector<DDim::value_type>{
              static_cast<int64_t>(output_image_shape["width"]),
              static_cast<int64_t>(output_image_shape["height"])}));
      cl::NDRange global_work_size =
          cl::NDRange{static_cast<size_t>(default_work_size[0]),
                      static_cast<size_t>(default_work_size[1]),
                      static_cast<size_t>(default_work_size[2])};

      cl_int status;
      status = kernel.setArg(0, *output_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(
          1, static_cast<size_t>(output_tensor_dims[1]));  // output_tensor_c
      CL_CHECK_FATAL(status);
      status = kernel.setArg(
          2, static_cast<size_t>(output_tensor_dims[3]));  // output_tensor_w
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, *input0_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, input0_tensor_c);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, *input1_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(6, input1_tensor_c);
      CL_CHECK_FATAL(status);
      if (inputs.size() >= 3) {
        status = kernel.setArg(7, *input2_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(8, input2_tensor_c);
        CL_CHECK_FATAL(status);
      }
      if (inputs.size() == 4) {
        status = kernel.setArg(9, *input3_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(10, input3_tensor_c);
        CL_CHECK_FATAL(status);
      }
      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "concat_mul") {  // inputs.size() > 3
      // TODO(ysh329): need to impl using buffer
      auto cur_axis_start_idx = 0;
      for (int i = 0; i < inputs.size(); i++) {
        auto* input = inputs[i];
        auto input_tensor_dims = input->dims();
        auto input_image_shape = InitImageDimInfoWith(input_tensor_dims);
        auto* input_image_p = input->data<half_t, cl::Image2D>();
        int input_tensor_w = input_tensor_dims[input_tensor_dims.size() - 1];

        global_work_size = cl::NDRange{
            static_cast<cl::size_type>(input_tensor_w),
            static_cast<cl::size_type>(input_image_shape["width"] /
                                       input_tensor_w),
            static_cast<cl::size_type>(input_image_shape["height"])};

#ifdef LITE_WITH_LOG
        VLOG(4) << "input_image_shape(w,h):" << input_image_shape["width"]
                << " " << input_image_shape["height"];
#endif

        arg_idx = 0;
        cl_int status = kernel.setArg(arg_idx, *input_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *output_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *output_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, flag_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, cur_axis_start_idx);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, output_tensor_c);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, output_tensor_w);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, input_tensor_w);
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
        cur_axis_start_idx += input->dims()[axis_];
      }
    }
  }

  std::string doc() { return "Concat using cl::Image, kFP16"; }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

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
