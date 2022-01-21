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

template <typename T>
void basic_gemm(const T* a,
                int M,
                int K,
                const T* b,
                int K_,
                int N,
                T* c,
                bool x_trans,
                bool y_trans) {
  EXPECT_TRUE(M > 0 && N > 0 && K > 0);
  EXPECT_TRUE(a && b && c);
  std::vector<T> transb(K_ * N);
  for (int i = 0; i < K_; ++i) {
    for (int j = 0; j < N; ++j) {
      transb[j * K_ + i] = y_trans ? b[i * N + j] : b[j * K_ + i];
    }
  }
  std::vector<T> transa(M * K);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      transa[j * M + i] = x_trans ? a[i * K + j] : a[j * M + i];
    }
  }
  if ((!x_trans) && (!y_trans)) {
    EXPECT_TRUE(K == K_);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        c[m * N + n] = 0.f;
        for (int k = 0; k < K; ++k) {
          c[m * N + n] += a[m * K + k] * b[k * N + n];
        }
      }
    }
  } else if ((!x_trans) && (y_trans)) {
    EXPECT_TRUE(K == N);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < K_; ++n) {
        c[m * K_ + n] = 0.f;
        for (int k = 0; k < K; ++k) {
          c[m * K_ + n] += a[m * K + k] * transb[k * K_ + n];
        }
      }
    }
  } else if ((x_trans) && (!y_trans)) {
    EXPECT_TRUE(K_ == M);
    for (int m = 0; m < K; ++m) {
      for (int n = 0; n < N; ++n) {
        c[m * N + n] = 0.f;
        for (int k = 0; k < K_; ++k) {
          c[m * N + n] += transa[m * M + k] * b[k * N + n];
        }
      }
    }
  } else if ((x_trans) && (y_trans)) {
    for (int m = 0; m < K; ++m) {
      for (int n = 0; n < K_; ++n) {
        c[m * K_ + n] = 0.f;
        for (int k = 0; k < M; ++k) {
          c[m * K_ + n] += transa[m * M + k] * transb[k * K_ + n];
        }
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
          int m,
          int k_x,
          int k_y,
          int n,
          bool x_transpose,
          bool y_transpose) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);

  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " m=" << m << " n=" << n
            << " k_x=" << k_x << " k_y=" << k_y;

  lite::Tensor x, y, out;
  operators::MatMulParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.transpose_X = x_transpose;
  param.transpose_Y = y_transpose;

  const DDim x_dim = DDim(std::vector<DDim::value_type>{m, k_x});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{k_y, n});
  const DDim out_dim = y_transpose ? DDim(std::vector<DDim::value_type>{m, k_y})
                                   : DDim(std::vector<DDim::value_type>{m, n});

  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  std::vector<float> x_source(x_dim.production());
  std::vector<float> y_source(y_dim.production());
  std::vector<float> out_gpu(out_dim.production());
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

  TargetWrapperCL::MemcpySync(out_gpu.data(),
                              out_data,
                              out_gpu.size() * sizeof(float),
                              IoDirection::DtoH);

  // run cpu ref
  std::vector<float> out_ref(out_dim.production());
  int x_inner = x_dim[x_dim.size() - 2] * x_dim[x_dim.size() - 1];
  int o_inner = out_dim[out_dim.size() - 2] * out_dim[out_dim.size() - 1];
  for (int i = 0; i < x_dim.count(0, x_dim.size() - 2); ++i) {
    basic_gemm<float>(x_source.data() + i * x_inner,
                      m,
                      k_x,
                      y_source.data(),
                      k_y,
                      n,
                      out_ref.data() + i * o_inner,
                      x_transpose,
                      y_transpose);
  }
#ifdef PRINT_RESULT
  PrintData("x", static_cast<float*>(x_source.data()), m, k_x);
  PrintData("w", static_cast<float*>(y_source.data()), k_y, n);
  PrintData("out_ref", static_cast<float*>(out_ref.data()), m, n);
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
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_gpu[i], out_ref[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_gpu[i], out_ref[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "  err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_gpu[i]
                   << "\t out_ref: " << out_ref[i];
      diff_cnt++;
      break;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " m=" << m << " n=" << n << " k_x=" << k_x;
}

TEST(matmul, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (int m : {1, 2, 5}) {
      for (int k : {1024, 3, 7, 10}) {
        for (int n : {357, 3, 9, 12}) {
          for (bool x_transpose : {false}) {
            for (bool y_transpose : {true, false}) {
              if (y_transpose)
                test(precision_type, m, k, n, k, x_transpose, y_transpose);
              else
                test(precision_type, m, k, k, n, x_transpose, y_transpose);
            }
          }
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(matmul, kOpenCL, kFloat, kNCHW, def);
