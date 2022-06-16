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

#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (2e-1)
#define FP16_RELATIVE_DIFF (2e-1)

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

void Transpose10(float* input, float* output, const int ih, const int iw) {
  for (auto h = 0; h < ih; h++) {
    for (auto w = 0; w < iw; w++) {
      output[w * ih + h] = input[h * iw + w];
    }
  }
}

void Transpose102(
    float* input, float* output, const int ic, const int ih, const int iw) {
  for (auto c = 0; c < ic; c++) {
    for (auto h = 0; h < ih; h++) {
      for (auto w = 0; w < iw; w++) {
        output[h * ic * iw + c * iw + w] = input[c * ih * iw + h * iw + w];
      }
    }
  }
}

static inline void TestWithKernel(
    const std::unique_ptr<paddle::lite::KernelBase>& kernel,
    const lite_api::CLPrecisionType p,
    const DDim& input_dim) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  bool input_persist = true;
  lite::Tensor input, input_h, output, output_h, input_persist_fp32;
  operators::TransposeParam param;

  if (input_persist) {
    if (fp16_flag) {
      param.x = &input_persist_fp32;
      param.output = &output_h;
    } else {
      param.x = &input_persist_fp32;
      param.output = &output;
    }
  } else {
    if (fp16_flag) {
      param.x = &input_h;
      param.output = &output_h;
    } else {
      param.x = &input;
      param.output = &output;
    }
  }

  DDim output_dim;
  DDim input_dim_full, output_dim_full;
  if (input_dim.size() == 2) {
    param.axis = std::vector<int>({1, 0});
    output_dim = lite::DDim{std::vector<int64_t>({input_dim[1], input_dim[0]})};
    input_dim_full =
        lite::DDim{std::vector<int64_t>({1, 1, input_dim[0], input_dim[1]})};
    output_dim_full =
        lite::DDim{std::vector<int64_t>({1, 1, output_dim[0], output_dim[1]})};
  } else if (input_dim.size() == 3) {
    param.axis = std::vector<int>({1, 0, 2});
    output_dim = lite::DDim{
        std::vector<int64_t>({input_dim[1], input_dim[0], input_dim[2]})};
    input_dim_full = lite::DDim{
        std::vector<int64_t>({1, input_dim[0], input_dim[1], input_dim[2]})};
    output_dim_full = lite::DDim{
        std::vector<int64_t>({1, output_dim[0], output_dim[1], output_dim[2]})};
  } else {
    param.axis = std::vector<int>({0, 2, 1, 3});
    output_dim = lite::DDim{std::vector<int64_t>(
        {input_dim[0], input_dim[2], input_dim[1], input_dim[3]})};
    input_dim_full = input_dim;
    output_dim_full = output_dim;
  }

  input.Resize(input_dim);
  input_h.Resize(input_dim);
  input_persist_fp32.Resize(input_dim);
  output.Resize(output_dim);
  output_h.Resize(output_dim);
  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  std::vector<float> x_source(input_dim.production());
  std::vector<half_t> x_source_half(input_dim.production());
  std::vector<float> output_source(output_dim.production());
  std::vector<float> output_half2float(output_dim.production());
  size_t x_size = input_dim.production() * sizeof(float);
  auto* input_persist_fp32_data = input_persist_fp32.mutable_data<float>();
  for (size_t i = 0; i < input_dim.production(); ++i) {
    x_source[i] = static_cast<int>(dist(engine));
    x_source_half[i] = Float2Half(x_source[i]);
    input_persist_fp32_data[i] = x_source[i];
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
  // for (int i = 0; i < 100; i++) {
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
  // }

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

  if (input_dim.size() == 4) {
    Transpose0213(x_source.data(),
                  output_source.data(),
                  input_dim[0],
                  input_dim[1],
                  input_dim[2],
                  input_dim[3]);
  } else if (input_dim.size() == 2) {
    Transpose10(
        x_source.data(), output_source.data(), input_dim[0], input_dim[1]);
  } else {
    Transpose102(x_source.data(),
                 output_source.data(),
                 input_dim[0],
                 input_dim[1],
                 input_dim[2]);
  }

  for (int eidx = 0; eidx < output_dim.production(); ++eidx) {
    output_half2float[eidx] = Half2Float(out_data_from_gpu_half.data()[eidx]);
  }

  PrintData("input",
            static_cast<float*>(x_source.data()),
            input_dim_full[0],
            input_dim_full[1],
            input_dim_full[2],
            input_dim_full[3]);
  PrintData("output",
            static_cast<float*>(output_source.data()),
            output_dim_full[0],
            output_dim_full[1],
            output_dim_full[2],
            output_dim_full[3]);
  PrintData("gpu",
            static_cast<float*>(out_data_from_gpu.data()),
            output_dim_full[0],
            output_dim_full[1],
            output_dim_full[2],
            output_dim_full[3]);
  PrintData("gpu_half",
            static_cast<float*>(output_half2float.data()),
            output_dim_full[0],
            output_dim_full[1],
            output_dim_full[2],
            output_dim_full[3]);
  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < output_dim.production(); i++) {
    auto out_gpu_data = out_data_from_gpu[i];
    if (fp16_flag) {
      out_gpu_data = output_half2float[i];
    }
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_gpu_data, output_source[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_gpu_data, output_source[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "   err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_gpu_data
                   << "\t out_ref: " << output_source[i];
      diff_cnt++;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << output_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " x_dim=" << input_dim;
  // check output data
  // index = 0;
  // for (auto n = 0; n < batch_size; n++) {
  //   for (auto h = 0; h < ih; h++) {
  //     for (auto c = 0; c < ic; c++) {
  //       for (auto w = 0; w < iw; w++) {
  //         auto input_index = n * ic * ih * iw + c * ih * iw + h * iw + w;
  //         auto input_value = x_source[input_index];
  //         float output_value = 0.f;
  //         if (fp16_flag) {
  //           output_value = Half2Float(out_data_from_gpu_half.data()[index]);
  //         } else {
  //           output_value = out_data_from_gpu[index];
  //         }
  //         auto abs_diff = abs(input_value - output_value);
  //         auto relative_diff = COMPUTE_RELATIVE_DIFF(input_value,
  //         output_value);
  //         EXPECT_EQ(
  //             (relative_diff <= FP16_MAX_DIFF) || (abs_diff <=
  //             FP16_MAX_DIFF),
  //             true);
  //         // if (relative_diff > FP16_MAX_DIFF){
  //         //   std::cout << "output_value: " << output_value << ";
  //         input_value:
  //         //   " << input_value << std::endl;
  //         // }
  //         index++;
  //       }
  //     }
  //   }
  // }
}

TEST(transpose_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create(
      "transpose", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP16,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (const auto x_dim : std::vector<std::vector<int64_t>>{{2, 3, 2}}) {
      int ndims = x_dim.size();
      TestWithKernel(kernel, precision_type, DDim(x_dim));
    }

    // Special case, such as large num
    // const auto x_dims = std::vector<int64_t>{1, 1000};
    // test(precision_type, DDim(x_dims), 1);
  }
}

// TEST(transpose2_opencl, compute) {
//   auto kernels = KernelRegistry::Global().Create(
//       "transpose2", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
//   ASSERT_FALSE(kernels.empty());
//   auto kernel = std::move(kernels.front());
//   // TestWithKernel(kernel);
// }

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(transpose, kOpenCL, kFP16, kNCHW, def);
