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

#include <gtest/gtest.h>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (2e-1)
#define FP16_RELATIVE_DIFF (2e-1)

namespace paddle {
namespace lite {

void PrintData(std::string name, float* a, DDim input_dim) {
  std::cout << "==== " << name << " ====" << std::endl;
  int input_dim_size = input_dim.size();
  DDim new_input_dim = DDim({1, 1, 1, 1});
  for (int i = 0; i < input_dim_size; i++) {
    new_input_dim[i + input_dim_size] = input_dim[i];
  }
  for (int n = 0; n < new_input_dim[0]; ++n) {
    for (int c = 0; c < new_input_dim[1]; ++c) {
      for (int h = 0; h < new_input_dim[2]; ++h) {
        for (int w = 0; w < new_input_dim[3]; ++w) {
          std::cout
              << " "
              << a[n * new_input_dim[1] * new_input_dim[2] * new_input_dim[3] +
                   c * new_input_dim[2] * new_input_dim[3] +
                   h * new_input_dim[3] + w];
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void reduce_sum_baseline(const float* x_data,
                         float* out_data,
                         DDim input_dim,
                         DDim output_dim) {
  int input_dim_size = input_dim.size();
  int output_dim_size = output_dim.size();
  DDim new_input_dim = DDim({1, 1, 1, 1});
  DDim new_output_dim = DDim({1, 1, 1, 1});
  for (int i = 0; i < input_dim_size; i++) {
    new_input_dim[i + input_dim_size] = input_dim[i];
  }
  for (int i = 0; i < output_dim_size; i++) {
    new_output_dim[i + output_dim_size] = output_dim[i];
  }
  float sum = 0.f;
  for (int n = 0; n < new_input_dim[0]; ++n) {
    for (int c = 0; c < new_input_dim[1]; ++c) {
      for (int h = 0; h < new_input_dim[2]; ++h) {
        sum = 0.f;
        for (int w = 0; w < new_input_dim[3]; ++w) {
          int src_p =
              n * new_input_dim[1] * new_input_dim[2] * new_input_dim[3] +
              c * new_input_dim[2] * new_input_dim[3] + h * new_input_dim[3] +
              w;
          sum += x_data[src_p];
        }
        int dst_p =
            n * new_output_dim[1] * new_output_dim[2] * new_output_dim[3] +
            c * new_output_dim[2] * new_output_dim[3] + h * new_output_dim[3];
        out_data[dst_p] = sum;
      }
    }
  }
}

void test(const lite_api::CLPrecisionType p, const DDim& input_dim) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " x_dim=" << input_dim;

  auto kernels = KernelRegistry::Global().Create(
      "reduce_sum", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  lite::Tensor input, input_h, output, output_h;
  operators::ReduceParam param;
  if (fp16_flag) {
    param.X = &input_h;
    param.Out = &output_h;
  } else {
    param.X = &input;
    param.Out = &output;
  }
  param.dim = {1};
  param.keep_dim = true;

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  DDim output_dim = input_dim;
  input.Resize(input_dim);
  input_h.Resize(input_dim);

  bool reduce_all = false;
  auto x_rank = input_dim.size();
  // lite::Tensor x, out;
  if (!param.dim.empty()) {
    for (size_t i = 0; i < param.dim.size(); i++) {
      if (param.dim[i] < 0) {
        param.dim[i] += x_rank;
      }
    }
  }

  std::stable_sort(param.dim.begin(), param.dim.end());
  if (param.dim.size() == 0) {
    reduce_all = true;
  }

  std::vector<int64_t> out_dims_shape;
  // DDim out_dims;
  if (reduce_all) {
    if (param.keep_dim) {
      for (size_t i = 0; i < input_dim.size(); i++) {
        out_dims_shape.push_back(1);
      }
    } else {
      out_dims_shape.push_back(1);
    }
  } else {
    for (size_t i = 0; i < input_dim.size(); i++) {
      out_dims_shape.push_back(input_dim[i]);
    }
    if (param.keep_dim) {
      for (size_t i = 0; i < param.dim.size(); ++i) {
        out_dims_shape[param.dim[i]] = 1L;
      }
    } else {
      int64_t kDelFlag = -2;
      for (size_t i = 0; i < param.dim.size(); ++i) {
        out_dims_shape[param.dim[i]] = kDelFlag;
      }
      out_dims_shape.erase(
          remove(out_dims_shape.begin(), out_dims_shape.end(), kDelFlag),
          out_dims_shape.end());
    }
    if (!param.keep_dim && out_dims_shape.empty()) {
      out_dims_shape.push_back(1);
    }
    output.Resize(DDim(out_dims_shape));
    output_h.Resize(DDim(out_dims_shape));
  }

  output_dim = DDim(out_dims_shape);

  // std::vector<float> x_cpu(x_dim.production());
  // std::vector<float> out_from_cpu(out_dim.production());
  // std::vector<float> out_from_gpu(out_dim.production());
  // fill_data_rand(x_cpu.data(), -1.f, 1.f, x_dim.production());

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
  // CLImageConverterFolder* folder_converter = new CLImageConverterFolder();
  // DDim x_image_shape = folder_converter->InitImageDimInfoWith(x_dim);
  // DDim out_image_shape = folder_converter->InitImageDimInfoWith(out_dim);
  // VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " <<
  // x_image_shape[1];
  // VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
  //         << out_image_shape[1];

  // const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  // std::vector<char> x_image_data(x_image_shape.production() * 4 *
  // dtype_size);
  // folder_converter->NCHWToImage(x_cpu.data(), x_image_data.data(), x_dim);
  // MUTABLE_DATA_GPU(&x, x_image_shape[0], x_image_shape[1],
  // x_image_data.data());
  // auto* out_image =
  //     MUTABLE_DATA_GPU(&out, out_image_shape[0], out_image_shape[1],
  //     nullptr);

  // run opencl kernel
  // for (int i = 0; i < 100; i++){
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

  // run cpu ref
  reduce_sum_baseline(
      x_source.data(), output_source.data(), input_dim, output_dim);

  for (int eidx = 0; eidx < output_dim.production(); ++eidx) {
    output_half2float[eidx] = Half2Float(out_data_from_gpu_half.data()[eidx]);
  }

  PrintData("input", static_cast<float*>(x_source.data()), input_dim);
  PrintData("output", static_cast<float*>(output_source.data()), output_dim);
  if (fp16_flag) {
    PrintData(
        "gpu_half", static_cast<float*>(output_half2float.data()), output_dim);
  } else {
    PrintData("gpu", static_cast<float*>(out_data_from_gpu.data()), output_dim);
  }

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
}

TEST(reduce_sum, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP16,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (const auto x_dim : std::vector<std::vector<int64_t>>{{3, 4}}) {
      int ndims = x_dim.size();
      test(precision_type, DDim(x_dim));
    }

    // Special case, such as large num
    // const auto input_dim = std::vector<int64_t>{1, 1000};
    // test(precision_type, DDim(input_dim), 1);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reduce_sum, kOpenCL, kFP16, kNCHW, def);
