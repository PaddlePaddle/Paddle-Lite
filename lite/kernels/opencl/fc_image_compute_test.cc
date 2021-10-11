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

#define FP32_ABS_DIFF (5e-4)
#define FP32_RELATIVE_DIFF (1e-3)
#define FP16_ABS_DIFF (8e-2)
#define FP16_RELATIVE_DIFF (8e-2)

// #define PRINT_RESULT

namespace paddle {
namespace lite {

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]
#define C(i, j) c[i * ldc + j]

template <typename T>
void gemm_bias(const T* a,
               const int M,
               const int K,
               const T* b,
               const int K_,
               const int N,
               T* biases,
               T* c) {
  EXPECT_TRUE(K_ == K && M > 0 && N > 0 && K > 0);
  EXPECT_TRUE(a && b && c);
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      C(m, n) = 0.0f;
      for (int k = 0; k < K; ++k) {
        C(m, n) += A(m, k) * B(k, n);
      }
    }
  }
  if (biases) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C(m, n) += biases[n];
      }
    }
  }
}

void PrintData(std::string name, float* a, const int rows, const int cols) {
  std::cout << "==== " << name << " ====" << std::endl;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << " " << a[r * cols + c];
    }
    std::cout << std::endl;
  }
}

void test(const lite_api::CLPrecisionType p,
          const bool bias_flag,
          const int m,
          const int n,
          const int k) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " bias_flag=" << bias_flag
            << " m=" << m << " n=" << n << " k=" << k;

  auto kernels = KernelRegistry::Global().Create(
      "fc", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  lite::Tensor x, w, bias, out;
  operators::FcParam param;
  param.input = &x;
  param.w = &w;
  param.bias = &bias;
  param.bias = bias_flag ? &bias : nullptr;
  param.output = &out;
  param.in_num_col_dims = 1;

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> fc_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(&(fc_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(fc_context));

  const DDim x_dim = DDim(std::vector<DDim::value_type>{m, k});
  const DDim w_dim = DDim(std::vector<DDim::value_type>{k, n});
  const DDim bias_dim = DDim(std::vector<DDim::value_type>{n});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{m, n});
  const DDim x_ext_dim = DDim(std::vector<DDim::value_type>{m, k, 1, 1});
  const DDim out_ext_dim = DDim(std::vector<DDim::value_type>{m, n, 1, 1});

  x.Resize(x_dim);
  w.Resize(w_dim);
  out.Resize(out_dim);

  std::vector<float> x_source(x_dim.production());
  std::vector<float> w_source(w_dim.production());
  std::vector<float> bias_source;
  std::vector<float> out_ref(out_dim.production());
  std::vector<float> out_gpu(out_dim.production());
  fill_data_rand(x_source.data(), -1.f, 1.f, x_source.size());
  fill_data_rand(w_source.data(), -1.f, 1.f, w_source.size());

  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(x_ext_dim);
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_ext_dim);
  VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " << x_image_shape[1];
  VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
          << out_image_shape[1];

  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  default_converter->NCHWToImage(
      x_source.data(), x_image_data.data(), x_ext_dim);
  MUTABLE_DATA_GPU(&x, x_image_shape[0], x_image_shape[1], x_image_data.data());
  auto* out_image =
      MUTABLE_DATA_GPU(&out, out_image_shape[0], out_image_shape[1], nullptr);

  w.Assign<float, lite::DDim, TARGET(kARM)>(w_source.data(), w_dim);

  if (bias_flag) {
    bias.Resize(bias_dim);
    bias_source.resize(bias_dim.production());
    fill_data_rand(bias_source.data(), -1.f, 1.f, bias_source.size());
    bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_source.data(), bias_dim);
  }

  // run opencl kernel
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  std::vector<char> out_image_data(out_image_shape.production() * 4 *
                                   dtype_size);  // 4 : RGBA
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_converter->ImageToNCHW(
      out_image_data.data(), out_gpu.data(), out_image_shape, out_ext_dim);

  // run cpu ref
  gemm_bias<float>(x_source.data(),
                   m,
                   k,
                   w_source.data(),
                   k,
                   n,
                   bias_source.data(),
                   out_ref.data());
#ifdef PRINT_RESULT
  PrintData("x", static_cast<float*>(x_source.data()), m, k);
  PrintData("w", static_cast<float*>(w_source.data()), k, n);
  if (bias_flag) {
    PrintData("bias", static_cast<float*>(bias_source.data()), 1, n);
  }
  PrintData("out_ref", static_cast<float*>(out_ref), m, n);
  PrintData("gpu_out", static_cast<float*>(out_gpu.data()), m, n);
#endif

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_gpu[i], out_ref[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_gpu[i], out_ref[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "   err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_gpu[i]
                   << "\t out_ref: " << out_ref[i];
      diff_cnt++;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " bias_flag=" << bias_flag << " m=" << m << " n=" << n
            << " k=" << k;
}

TEST(fc, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (const bool bias_flag : {false, true}) {
      for (auto m = 1; m <= 2; m++) {
        for (auto n = 1; n <= 9; n += 2) {
          for (auto k = 1; k <= 9; k += 2) {
            test(precision_type, bias_flag, m, n, k);
          }
        }
      }

      // Special case, such as large n or k
      // test(precision_type, bias_flag, 1, 1000, 1024);
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kOpenCL, kFP16, kImageFolder, image2d);
