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
#include "lite/tests/utils/fill_data.h"

#define FP32_ABS_DIFF (1e-1)
#define FP32_RELATIVE_DIFF (8e-2)
#define FP16_ABS_DIFF (8e-2)
#define FP16_RELATIVE_DIFF (8e-2)

namespace paddle {
namespace lite {

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]
#define C(i, j) c[i * ldc + j]
template <typename T>
void basic_gemm(const T* a,
                const int M,
                const int K,
                const T* b,
                const int K_,
                const int N,
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
          const int m,
          const int n,
          const int k) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " m=" << m << " n=" << n
            << " k=" << k;

  lite::Tensor x, y, out;
  operators::MatMulParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.transpose_X = false;
  param.transpose_Y = false;

  const DDim x_dim = DDim(std::vector<DDim::value_type>{m, k});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{k, n});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{m, n});
  const DDim x_ext_dim = DDim(std::vector<DDim::value_type>{m, k, 1, 1});
  const DDim out_ext_dim = DDim(std::vector<DDim::value_type>{m, n, 1, 1});

  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  std::vector<float> x_source(x_dim.production());
  std::vector<float> y_source(y_dim.production());
  fill_data_rand(x_source.data(), -1.f, 1.f, x_source.size());
  fill_data_rand(y_source.data(), -1.f, 1.f, y_source.size());

  auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* y_data = y.mutable_data<float>();
  auto* out_data = out.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  TargetWrapperCL::MemcpySync(x_data,
                              x_source.data(),
                              x_source.size() * sizeof(float),
                              IoDirection::HtoD);
  y.Assign<float, lite::DDim, TARGET(kHost)>(y_source.data(), y_dim);

  // set kernel
  auto kernels = KernelRegistry::Global().Create(
      "matmul", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);

  // set context
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  std::unique_ptr<KernelContext> matmul_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(matmul_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(matmul_context));

  // run opencl kernel
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();

  std::vector<float> out_gpu(out_dim.production());
  TargetWrapperCL::MemcpySync(out_gpu.data(),
                              out_data,
                              out_gpu.size() * sizeof(float),
                              IoDirection::DtoH);

  // run cpu ref
  lite::Tensor out_ref;
  out_ref.Resize(out_dim);
  auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
  basic_gemm<float>(x_source.data(), m, k, y_source.data(), k, n, out_ref_data);

#ifdef PRINT_RESULT
  PrintData("x", static_cast<float*>(x_source.data()), m, k);
  PrintData("w", static_cast<float*>(w_source.data()), k, n);
  PrintData("out_ref", static_cast<float*>(out_ref_data), m, n);
  PrintData("gpu_out", static_cast<float*>(out_gpu.data()), m, n);
#endif

  LOG(INFO) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  LOG(INFO) << "relative_diff_thres: " << relative_diff_thres
            << "abs_diff_thres: " << abs_diff_thres;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_gpu[i], out_ref_data[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_gpu[i], out_ref_data[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "  err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_gpu[i]
                   << "\t out_ref: " << out_ref_data[i];
      diff_cnt++;
      break;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " m=" << m << " n=" << n << " k=" << k;
  getchar();
}

TEST(matmul, compute_basic) {
  const auto precision_type = lite_api::CLPrecisionType::CL_PRECISION_FP32;
  int m = 1, k = 1024;
  std::vector<int> vec_n = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1000};
  for (auto n = 0; n < vec_n.size(); n++) {
    test(precision_type, m, vec_n[n], k);
  }

  // Special case, such as large n or k
  // test(precision_type, 1, 1000, 1024);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(matmul, kOpenCL, kFloat, kNCHW, def);
