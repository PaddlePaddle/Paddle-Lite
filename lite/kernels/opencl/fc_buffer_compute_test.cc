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

#define FP16_MAX_DIFF (1e-2)

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

// #define PRINT_RESULT
// #define LOOP_TEST
TEST(fc, compute) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

#ifdef LOOP_TEST
  for (int m = 1; m < 4; m += 1) {
    for (int k = 1; k < 4; k += 1) {
      for (int n = 1; n < 4; n += 1) {
#else
#if 0
  const int m = 1;
  const int k = 1024;
  const int n = 1000;
#else
  // m,k,n:2,3,1
  //       1,2,3
  //       2,1,3
  //       1,2,3
  const int m = 1;
  const int k = 2;
  const int n = 3;
#endif
#endif
        LOG(INFO) << "m=" << m << " n=" << n << " k=" << k;

        auto kernels = KernelRegistry::Global().Create(
            "fc", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
        ASSERT_FALSE(kernels.empty());
        auto kernel = std::move(kernels.front());

        lite::Tensor x, w, bias, out, out_ref;
        operators::FcParam param;
        param.input = &x;
        param.w = &w;
        param.bias = &bias;
        param.output = &out;
        param.in_num_col_dims = 1;

        kernel->SetParam(param);
        std::unique_ptr<KernelContext> fc_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(fc_context->As<OpenCLContext>()));
        kernel->SetContext(std::move(fc_context));

        const DDim x_dim = DDim(std::vector<DDim::value_type>{m, k});
        const DDim w_dim = DDim(std::vector<DDim::value_type>{k, n});
        const DDim bias_dim = DDim(std::vector<DDim::value_type>{n});
        const DDim out_dim = DDim(std::vector<DDim::value_type>{m, n});

        x.Resize(x_dim);
        w.Resize(w_dim);
        bias.Resize(bias_dim);
        out.Resize(out_dim);
        out_ref.Resize(out_dim);

        VLOG(2) << "out.dims():" << out.dims() << ", out_dim:" << out_dim;

        auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto* w_data = w.mutable_data<float>();
        auto* bias_data = bias.mutable_data<float>();
        auto* out_data = out.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

        std::default_random_engine engine;
        std::uniform_real_distribution<float> dist(-5, 5);

        std::vector<float> x_source(x_dim.production());
        std::vector<float> w_source(w_dim.production());
        std::vector<float> bias_source(bias_dim.production());

        size_t x_size = x_dim.production() * sizeof(float);
        size_t w_size = w_dim.production() * sizeof(float);
        size_t bias_size = bias_dim.production() * sizeof(float);
        size_t out_size = out_dim.production() * sizeof(float);

        for (size_t i = 0; i < x_dim.production(); ++i) {
          x_source[i] = static_cast<int>(dist(engine));
        }
        for (size_t i = 0; i < w_dim.production(); ++i) {
          w_source[i] = static_cast<int>(dist(engine));
          w_data[i] = w_source[i];
        }
        for (size_t i = 0; i < bias_dim.production(); ++i) {
          bias_source[i] = 10;  // static_cast<int>(dist(engine));
          bias_data[i] = 10;
        }

        TargetWrapperCL::MemcpySync(
            x_data, x_source.data(), x_size, IoDirection::HtoD);

        // run opencl kernel
        kernel->Launch();
        CLRuntime::Global()->command_queue().finish();

#if 0  // NOTE(ysh329): note event
        auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
        auto* out_ptr = param.output->data<float, cl::Buffer>();
        auto it = wait_list->find(out_ptr);
        if (it != wait_list->end()) {
          VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
          auto& event = *(it->second);
          event.wait();
        CLRuntime::Global()->command_queue().finish();
          double start_nanos =
              event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
          double stop_nanos =
              event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
          double elapsed_micros = (stop_nanos - start_nanos) / 1000.0;
          LOG(INFO) << "Kernel Run Cost Time: " << elapsed_micros << " us.";
        } else {
          LOG(FATAL)
              << "Could not find the sync event for the target cl tensor.";
        }
#endif

        std::vector<float> out_data_from_gpu(out_dim.production());
        TargetWrapperCL::MemcpySync(out_data_from_gpu.data(),
                                    out_data,
                                    out_data_from_gpu.size() * sizeof(float),
                                    IoDirection::DtoH);

        // run cpu ref
        auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
        gemm_bias<float>(x_source.data(),
                         m,
                         k,
                         w_source.data(),
                         k,
                         n,
                         bias_source.data(),
                         out_ref_data);
#ifdef PRINT_RESULT
        PrintData("x", static_cast<float*>(x_source.data()), m, k);
        PrintData("w", static_cast<float*>(w_source.data()), k, n);
        PrintData("bias", static_cast<float*>(bias_source.data()), 1, n);
        PrintData("out_ref_data", static_cast<float*>(out_ref_data), m, n);
        PrintData(
            "gpu_out", static_cast<float*>(out_data_from_gpu.data()), m, n);
#endif

        for (int eidx = 0; eidx < out_dim.production(); ++eidx) {
          auto abs_diff = COMPUTE_ABS_DIFF(out_ref_data[eidx],
                                           out_data_from_gpu.data()[eidx]);
          auto relative_diff = COMPUTE_RELATIVE_DIFF(
              out_ref_data[eidx], out_data_from_gpu.data()[eidx]);
          EXPECT_EQ(
              (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
          if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
            LOG(FATAL) << "error idx:" << eidx << ", out_ref_data[" << eidx
                       << "]:" << out_ref_data[eidx]
                       << ", out_data_from_gpu.data()[" << eidx
                       << "]:" << out_data_from_gpu.data()[eidx]
                       << " abs_diff:" << abs_diff
                       << " relative_diff:" << relative_diff
                       << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
          }
        }

#ifdef LOOP_TEST
      }  // n
    }    // k
  }      // m
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kOpenCL, kFloat, kNCHW, def);
