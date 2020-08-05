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

    auto input_num = concat_param_->x.size();
    auto* output = concat_param_->output;
    auto output_dims_size = output->dims().size();
    auto axis = concat_param_->axis;
    if (output_dims_size < 4) {
      if (output_dims_size - axis == 1) {
        kernel_func_name_ = "concatByW";
      } else {
        kernel_func_name_ = "concatByH";
      }
    } else if (output_dims_size == 4) {  // output->dims.size() == 4
      if (input_num == 2) {
        kernel_func_name_ = "concatByCWith2Inputs";
      } else if (input_num == 3) {
        kernel_func_name_ = "concatByCWith3Inputs";
      } else if (input_num == 4) {
        kernel_func_name_ = "concatByCWith4Inputs";
      } else {
        LOG(FATAL) << "Unsupported input tensors number:" << input_num << ".";
      }
    } else {  // output->dims.size() > 4
      LOG(FATAL) << "Unsupported output dims " << output->dims()
                 << ", whose dims.size() is bigger than 4.";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/concat_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto output_tensor_dims = concat_param_->output->dims();
    auto output_image_shape = InitImageDimInfoWith(output_tensor_dims);
    auto output_image_p =
        concat_param_->output->mutable_data<half_t, cl::Image2D>(
            output_image_shape["width"], output_image_shape["height"]);

    auto inputs = concat_param_->x;
    auto axis = concat_param_->axis;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    if (kernel_func_name_ == "concatByW" || kernel_func_name_ == "concatByH") {
      auto output_tensor_w = output_tensor_dims[output_tensor_dims.size() - 1];
      if (output_tensor_dims.size() - axis == 1) {
        for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
          auto* input = inputs[input_idx];
          auto input_tensor_dims = input->dims();
          auto input_image_shape = InitImageDimInfoWith(input_tensor_dims);
          auto input_tensor_w = input_tensor_dims[input_tensor_dims.size() - 1];
          auto* input_image_p = input->data<half_t, cl::Image2D>();

          size_t input_tensor_pre_w = 0;
          for (size_t ii_idx = 0; ii_idx < input_idx; ++ii_idx) {
            auto input_tensor_dims = inputs[ii_idx]->dims();
            input_tensor_pre_w +=
                input_tensor_dims[input_tensor_dims.size() - 1];
          }

          int input_special_w = input_tensor_dims[output_tensor_dims.size() -
                                                  2];  // not a good var name

          const std::vector<size_t>& default_work_size = DefaultWorkSize(
              input_tensor_dims,
              DDim(std::vector<DDim::value_type>{
                  static_cast<int64_t>(input_image_shape["width"]),
                  static_cast<int64_t>(input_image_shape["height"])}));
          cl::NDRange global_work_size =
              cl::NDRange{static_cast<size_t>(default_work_size[0]),
                          static_cast<size_t>(default_work_size[1]),
                          static_cast<size_t>(default_work_size[2])};
          cl_int status;
          status = kernel.setArg(0, *input_image_p);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(1, *output_image_p);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(2, input_special_w);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(3, input_tensor_pre_w);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(4, output_tensor_w);
          CL_CHECK_FATAL(status);

          status = EnqueueNDRangeKernel(context,
                                        kernel,
                                        cl::NullRange,
                                        global_work_size,
                                        cl::NullRange,
                                        nullptr,
                                        event_);
          CL_CHECK_FATAL(status);
        }
      } else {
        size_t output_image_height_start = 0;  // output image height start
        for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
          auto* input = inputs[input_idx];
          auto input_tensor_dims = input->dims();
          auto input_image_shape = InitImageDimInfoWith(input_tensor_dims);
          auto input_tensor_w = input_tensor_dims[input_tensor_dims.size() - 1];
          auto* input_image_p = input->data<half_t, cl::Image2D>();

          const std::vector<size_t>& default_work_size = DefaultWorkSize(
              input_tensor_dims,
              DDim(std::vector<DDim::value_type>{
                  static_cast<int64_t>(input_image_shape["width"]),
                  static_cast<int64_t>(input_image_shape["height"])}));
          cl::NDRange global_work_size =
              cl::NDRange{static_cast<size_t>(default_work_size[0]),
                          static_cast<size_t>(default_work_size[1]),
                          static_cast<size_t>(default_work_size[2])};

          cl_int status;
          status = kernel.setArg(0, *input_image_p);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(1, *output_image_p);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(2, output_tensor_w);
          CL_CHECK_FATAL(status);
          status = kernel.setArg(3, output_image_height_start);
          CL_CHECK_FATAL(status);

          status = EnqueueNDRangeKernel(context,
                                        kernel,
                                        cl::NullRange,
                                        global_work_size,
                                        cl::NullRange,
                                        nullptr,
                                        event_);
          CL_CHECK_FATAL(status);

          // compute new output_image_height_start
          if (output_tensor_dims.size() == 3) {
            output_image_height_start += input_tensor_dims[1];
          } else if (output_tensor_dims.size() == 2) {
            output_image_height_start += input_tensor_dims[0];
          }
        }
      }
    } else if (kernel_func_name_ == "concatByCWith2Inputs" ||
               kernel_func_name_ == "concatByCWith3Inputs" ||
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

#ifdef LITE_WITH_LOG
      LOG(INFO) << "output_tensor_dims[1]:" << output_tensor_dims[1];
      LOG(INFO) << "output_tensor_dims[3]:" << output_tensor_dims[3];
      LOG(INFO) << "input0_image_p:" << input0_image_p;
      LOG(INFO) << "input0_tensor_c:" << input0_tensor_c;
      LOG(INFO) << "input1_image_p:" << input1_image_p;
      LOG(INFO) << "input1_tensor_c:" << input1_tensor_c;
      LOG(INFO) << "input2_image_p:" << input2_image_p;
      LOG(INFO) << "input2_tensor_c:" << input2_tensor_c;
      LOG(INFO) << "input3_image_p:" << input3_image_p;
      LOG(INFO) << "input3_tensor_c:" << input3_tensor_c;
#endif

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
    } else {
      LOG(FATAL) << "Unsupported kernel func name: " << kernel_func_name_;
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
