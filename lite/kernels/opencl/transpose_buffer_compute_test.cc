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

#include <gtest/gtest.h>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/operators/reshape_op.h"
#include "lite/utils/log/logging.h"

#define FP16_MAX_DIFF (3e-4)

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void PrintData(std::string name,
               float* a,
               const int in,
               const int ic,
               const int ih,
               const int iw) {
  std::cout << "==== " << name << " ====" << std::endl;
  for (int n = 0; n < in; ++n) {
    for (int c = 0; c < ic; ++c) {
      for (int h = 0; h < ih; ++h) {
        for (int w = 0; w < iw; ++w) {
          std::cout << " " << a[n * ic * ih * iw + c * ih * iw + h * iw + w];
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void Transpose0213(float* input,
                   float* output,
                   const int batch_size,
                   const int ic,
                   const int ih,
                   const int iw) {
  auto cxw = ic * iw;
  auto hxcxw = ih * cxw;
  for (auto n = 0; n < batch_size; n++) {
    for (auto h = 0; h < ih; h++) {
      for (auto c = 0; c < ic; c++) {
        for (auto w = 0; w < iw; w++) {
          auto input_index = n * ic * ih * iw + c * ih * iw + h * iw + w;
          auto output_index = n * hxcxw + h * cxw + c * iw + w;
          auto input_value = input[input_index];
          output[output_index] = input_value;
        }
      }
    }
  }
}

static inline void TestWithKernel(
    const std::unique_ptr<paddle::lite::KernelBase>& kernel) {
  int64_t batch_size = 1;
  int64_t ic = 4;
  int64_t ih = 128;
  int64_t iw = 64;

  int64_t oc = 128;
  int64_t oh = 4;
  int64_t ow = 64;

  lite_api::CLPrecisionType p = lite_api::CLPrecisionType::CL_PRECISION_FP16;
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);

  lite::Tensor input, input_h, output, output_h;
  operators::TransposeParam param;

  if (fp16_flag) {
    param.x = &input_h;
    param.output = &output_h;
  } else {
    param.x = &input;
    param.output = &output;
  }
  param.axis = std::vector<int>({0, 2, 1, 3});
  const DDim input_dim =
      lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};
  input.Resize(input_dim);
  input_h.Resize(input_dim);
  const DDim output_dim =
      lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
  output.Resize(output_dim);
  output_h.Resize(output_dim);
  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);

  // std::vector<float> input_v(batch_size * ic * ih * iw);

  LOG(INFO) << "gen input ...";

  // float* input_v_data = &input_v[0];
  auto index = 0;
  // for (auto& i : input_v) {
  //   i = index++;
  // }
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  std::vector<float> x_source(input_dim.production());
  std::vector<half_t> x_source_half(input_dim.production());
  std::vector<float> output_source(output_dim.production());
  std::vector<float> output_half2float(output_dim.production());
  size_t x_size = input_dim.production() * sizeof(float);
  for (size_t i = 0; i < input_dim.production(); ++i) {
    x_source[i] = static_cast<int>(dist(engine));
    x_source_half[i] = Float2Half(x_source[i]);
  }

  auto* x_data = input.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* x_data_h = input_h.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL));
  auto* out_data = output.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* out_data_h = output_h.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL));
  if (fp16_flag) {
    x_size = input_dim.production() * sizeof(half_t);
    TargetWrapperCL::MemcpySync(
        x_data_h, x_source_half.data(), x_size, IoDirection::HtoD);
  } else {
    TargetWrapperCL::MemcpySync(
        x_data, x_source.data(), x_size, IoDirection::HtoD);
  }

  LOG(INFO) << "kernel context ...";
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  std::unique_ptr<KernelContext> transpose_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(transpose_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(transpose_context));

  LOG(INFO) << "kernel launch ...";
  for (int i = 0; i < 100; i++) {
    kernel->Launch();
    CLRuntime::Global()->command_queue().finish();
  }

  std::vector<float> out_data_from_gpu(output_dim.production());
  std::vector<half_t> out_data_from_gpu_half(output_dim.production());
  if (fp16_flag) {
    TargetWrapperCL::MemcpySync(out_data_from_gpu_half.data(),
                                out_data_h,
                                out_data_from_gpu_half.size() * sizeof(half_t),
                                IoDirection::DtoH);
  } else {
    TargetWrapperCL::MemcpySync(out_data_from_gpu.data(),
                                out_data,
                                out_data_from_gpu.size() * sizeof(float),
                                IoDirection::DtoH);
  }

  Transpose0213(x_source.data(), output_source.data(), batch_size, ic, ih, iw);

  for (int eidx = 0; eidx < output_dim.production(); ++eidx) {
    output_half2float[eidx] = Half2Float(out_data_from_gpu_half.data()[eidx]);
  }

  // PrintData("input", static_cast<float*>(x_source.data()), batch_size, ic,
  // ih, iw);
  // PrintData("output", static_cast<float*>(output_source.data()), batch_size,
  // oc, oh, ow);
  // PrintData("gpu", static_cast<float*>(out_data_from_gpu.data()), batch_size,
  // oc, oh, ow);
  // PrintData("gpu_half", static_cast<float*>(output_half2float.data()),
  // batch_size, oc, oh, ow);
  // check output data
  index = 0;
  for (auto n = 0; n < batch_size; n++) {
    for (auto h = 0; h < ih; h++) {
      for (auto c = 0; c < ic; c++) {
        for (auto w = 0; w < iw; w++) {
          auto input_index = n * ic * ih * iw + c * ih * iw + h * iw + w;
          auto input_value = x_source[input_index];
          float output_value = 0.f;
          if (fp16_flag) {
            output_value = Half2Float(out_data_from_gpu_half.data()[index]);
          } else {
            output_value = out_data_from_gpu[index];
          }
          auto abs_diff = abs(input_value - output_value);
          auto relative_diff = COMPUTE_RELATIVE_DIFF(input_value, output_value);
          EXPECT_EQ(
              (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
          // if (relative_diff > FP16_MAX_DIFF){
          //   std::cout << "output_value: " << output_value << "; input_value:
          //   " << input_value << std::endl;
          // }
          index++;
        }
      }
    }
  }
}

TEST(transpose_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create(
      "transpose", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  TestWithKernel(kernel);
}

TEST(transpose2_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create(
      "transpose2", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  TestWithKernel(kernel);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(transpose, kOpenCL, kFloat, kNCHW, def);
