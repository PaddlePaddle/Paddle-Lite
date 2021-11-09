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
#include <vector>
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
          DDim in_x_dim,
          DDim in_y_dim,
          bool x_transpose = false,
          bool y_transpose = false) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " x_dim:= " << in_x_dim
            << " y_dim:= " << in_y_dim;
  int m, k_x, k_y, n;
  DDim x_dim, y_dim, out_dim;
  lite::Tensor x, y, out;
  operators::MatMulParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.transpose_X = x_transpose;
  param.transpose_Y = y_transpose;

  if (in_x_dim.size() <= 2 && in_y_dim.size() <= 2) {
    if (in_x_dim.size() == 2 && in_y_dim.size() == 2) {
      m = in_x_dim[0], k_x = in_x_dim[1];
      n = in_y_dim[1], k_y = in_y_dim[0];
    } else if (in_x_dim.size() == 1 && in_y_dim.size() == 1 &&
               in_x_dim[0] == in_y_dim[0]) {
      CHECK_EQ(x_transpose, false) << "unsupported when x_transpose is true";
      CHECK_EQ(y_transpose, false) << "unsupported when x_transpose is true";
      m = 1, n = 1;
      k_x = in_x_dim[0], k_y = in_y_dim[0];
    } else if (in_x_dim.size() == 1 && in_y_dim.size() == 1 &&
               in_x_dim[0] != in_y_dim[0]) {
      CHECK_EQ(x_transpose, true) << "unsupported when x_transpose is false";
      CHECK_EQ(y_transpose, true) << "unsupported when x_transpose is false";
      m = 1, n = 1;
      k_x = in_x_dim[0], k_y = in_y_dim[0];
    } else {
      LOG(FATAL) << "unsupported input case.";
    }
    x_dim = DDim(std::vector<DDim::value_type>{m, k_x});
    y_dim = DDim(std::vector<DDim::value_type>{k_y, n});
    int out_m = x_transpose ? k_x : m;
    int out_n = y_transpose ? k_y : n;
    out_dim = DDim(std::vector<DDim::value_type>{out_m, out_n});
  } else if (in_x_dim.size() > 2 && in_y_dim.size() >= 2) {
    out_dim = in_x_dim;
    x_dim = in_x_dim, y_dim = in_y_dim;
    m = in_x_dim[in_x_dim.size() - 2];
    n = in_y_dim[in_y_dim.size() - 1];
    k_x = in_x_dim[in_x_dim.size() - 1];
    k_y = in_y_dim[in_y_dim.size() - 2];
    if ((!x_transpose) && (!y_transpose)) {
      out_dim[in_x_dim.size() - 2] = in_x_dim[in_x_dim.size() - 2];
      out_dim[in_x_dim.size() - 1] = in_y_dim[in_y_dim.size() - 1];
    } else if ((!x_transpose) && y_transpose) {
      out_dim[in_x_dim.size() - 2] = in_x_dim[in_x_dim.size() - 2];
      out_dim[in_x_dim.size() - 1] = in_y_dim[in_y_dim.size() - 2];
    } else if (x_transpose && (!y_transpose)) {
      out_dim[in_x_dim.size() - 2] = in_x_dim[in_x_dim.size() - 1];
      out_dim[in_x_dim.size() - 1] = in_y_dim[in_y_dim.size() - 1];
    } else {
      out_dim[in_x_dim.size() - 2] = in_x_dim[in_x_dim.size() - 1];
      out_dim[in_x_dim.size() - 1] = in_y_dim[in_y_dim.size() - 2];
    }
  } else if (in_x_dim.size() > 2 && in_y_dim.size() == 1) {
    CHECK_EQ(in_x_dim[in_x_dim.size() - 1], in_y_dim[0])
        << "not supported x_dims(" << in_x_dim << ") and y_dims(" << in_y_dim
        << ")";
    x_dim = in_x_dim,
    y_dim = DDim(std::vector<DDim::value_type>{in_y_dim[0], 1});
    m = 1;
    n = 1;
    k_x = in_x_dim[in_x_dim.size() - 1], k_y = in_y_dim[0];
    if (in_x_dim.size() == 4) {
      out_dim = DDim(
          std::vector<DDim::value_type>{in_x_dim[0], in_x_dim[1], in_x_dim[2]});
    } else if (in_x_dim.size() == 3) {
      out_dim = DDim(std::vector<DDim::value_type>{in_x_dim[0], in_x_dim[1]});
    }
  } else {
    LOG(FATAL) << "unsupported input case.";
  }

  x.Resize(in_x_dim);
  y.Resize(in_y_dim);
  out.Resize(out_dim);

  std::vector<float> x_source(x_dim.production());
  std::vector<float> y_source(y_dim.production());
  std::vector<float> out_gpu(out_dim.production());

  fill_data_rand(x_source.data(), -1.f, 1.f, x_source.size());
  fill_data_rand(y_source.data(), -1.f, 1.f, y_source.size());

  CLImageConverterFolder* folder_converter = new CLImageConverterFolder();
  DDim x_image_shape = folder_converter->InitImageDimInfoWith(in_x_dim);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  folder_converter->NCHWToImage(x_source.data(), x_image_data.data(), in_x_dim);
  MUTABLE_DATA_GPU(&x, x_image_shape[0], x_image_shape[1], x_image_data.data());
  LOG(INFO) << "x_img_shape = " << x_image_shape[0] << " " << x_image_shape[1];

  DDim out_image_shape = folder_converter->InitImageDimInfoWith(out_dim);
  auto* out_image =
      MUTABLE_DATA_GPU(&out, out_image_shape[0], out_image_shape[1], nullptr);
  LOG(INFO) << "out_img_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  y.Assign<float, lite::DDim, TARGET(kHost)>(y_source.data(), in_y_dim);

  // set kernel
  auto kernels = KernelRegistry::Global().Create(
      "matmul", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder));
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
  folder_converter->ImageToNCHW(
      out_image_data.data(), out_gpu.data(), out_image_shape, out_dim);

  // run cpu ref
  std::vector<float> out_ref(out_dim.production());
  if (in_x_dim.size() > 2 && in_y_dim.size() >= 2) {
    int x_inner = x_dim[x_dim.size() - 2] * x_dim[x_dim.size() - 1];
    if (in_y_dim.size() > 2) {
      int y_inner = y_dim[y_dim.size() - 2] * y_dim[y_dim.size() - 1];
      int o_inner = out_dim[out_dim.size() - 2] * out_dim[out_dim.size() - 1];
      int batch = x_dim.count(0, x_dim.size() - 2);
      for (int i = 0; i < batch; ++i) {
        basic_gemm<float>(x_source.data() + i * x_inner,
                          m,
                          k_x,
                          y_source.data() + i * y_inner,
                          k_y,
                          n,
                          out_ref.data() + i * o_inner,
                          x_transpose,
                          y_transpose);
      }
    } else {  // in_y_dim.size() = 2
      LOG(INFO) << "matmul: x_dim.size() > 2 && y_dim.size() == 2";
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
    }
  } else if (in_x_dim.size() > 2 && in_y_dim.size() == 1) {
    LOG(INFO) << "matmul: x_dim.size() > 2 && y_dim.size() == 1";
    int batch = x_dim.count(0, x_dim.size() - 1);
    for (int i = 0; i < batch; ++i) {
      basic_gemm<float>(x_source.data() + i * y_dim[0],
                        m,
                        k_x,
                        y_source.data(),
                        k_y,
                        n,
                        out_ref.data() + i,
                        x_transpose,
                        y_transpose);
    }
  } else {
    LOG(INFO) << "in_x_dim.size() <= 2";
    basic_gemm<float>(x_source.data(),
                      m,
                      k_x,
                      y_source.data(),
                      k_y,
                      n,
                      out_ref.data(),
                      x_transpose,
                      y_transpose);
  }
#ifdef PRINT_RESULT
  PrintData("x", static_cast<float*>(x_source.data()), m, k);
  PrintData("y", static_cast<float*>(y_source.data()), k, n);
  PrintData("out_ref", static_cast<float*>(out_ref), m, n);
  PrintData("gpu_out", static_cast<float*>(out_gpu.data()), m, n);
#endif

  LOG(INFO) << "output_data vs output_ref_data";
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
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "  err idx: " << i
                   << "\t out_ins: " << out_gpu[i]
                   << "\t out_ref: " << out_ref[i];
      diff_cnt++;
      break;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  auto gpu_version = CLRuntime::Global()->device().getInfo<CL_DEVICE_VERSION>();
  LOG(INFO) << "GPU_DEVICE_VERSION: " << gpu_version;
  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " m=" << m << " k_x=" << k_x << " k_y=" << k_y << " n=" << n;
}

// TEST(matmul, performace_test) {
//   for (const auto precision_type :
//        {lite_api::CLPrecisionType::CL_PRECISION_FP32,
//         lite_api::CLPrecisionType::CL_PRECISION_FP16
//         }) {
//     int m = 1, k = 1024;
//     std::vector<int> vec_n = {1, 2, 4, 8, 16, 32, 64,
//                               128, 256, 512, 1024, 1000};
//     for (auto n = 0; n < vec_n.size(); n++) {
//         test(precision_type, DDim(std::vector<DDim::value_type>{1, 1024}),
//                 DDim(std::vector<DDim::value_type>{1024, vec_n[n]}), false,
//                 false);
//         test(precision_type, DDim(std::vector<DDim::value_type>{vec_n[n],
//         1024}),
//                 DDim(std::vector<DDim::value_type>{1024, 1024}), false,
//                 false);
//     }
//   }
// }

TEST(matmul, compute_full) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    // dim 2x2
    for (int m : {1, 2, 8}) {
      for (int k : {1, 3, 5}) {
        for (int n : {1, 4, 6}) {
          test(precision_type,
               DDim(std::vector<DDim::value_type>{m, k}),
               DDim(std::vector<DDim::value_type>{k, n}),
               false,
               false);
        }
      }
    }

    // dim 2x2 ytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 2}),
         DDim(std::vector<DDim::value_type>{3, 2}),
         false,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{2, 4}),
         DDim(std::vector<DDim::value_type>{3, 4}),
         false,
         true);

    // dim 1x1
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3}),
         DDim(std::vector<DDim::value_type>{3}),
         false,
         false);

    // dim 1x1 xytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3}),
         DDim(std::vector<DDim::value_type>{5}),
         true,
         true);

    // dim 2x2 xtranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4}),
         DDim(std::vector<DDim::value_type>{3, 2}),
         true,
         false);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{2, 5}),
         DDim(std::vector<DDim::value_type>{2, 1}),
         true,
         false);

    // dim 2x2 xytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{6, 2}),
         DDim(std::vector<DDim::value_type>{3, 6}),
         true,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3}),
         DDim(std::vector<DDim::value_type>{1, 5}),
         true,
         true);

    // dim nxn
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 6, 2}),
         DDim(std::vector<DDim::value_type>{3, 4, 2, 5}),
         false,
         false);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3, 4}),
         DDim(std::vector<DDim::value_type>{5, 4, 6}),
         false,
         false);

    // dim nxn ytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 6, 2}),
         DDim(std::vector<DDim::value_type>{3, 4, 5, 2}),
         false,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3, 4}),
         DDim(std::vector<DDim::value_type>{5, 6, 4}),
         false,
         true);

    // dim nxn xtranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 2, 6}),
         DDim(std::vector<DDim::value_type>{3, 4, 2, 5}),
         true,
         false);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 4, 2}),
         DDim(std::vector<DDim::value_type>{5, 4, 6}),
         true,
         false);

    // dim nxn xytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 2, 6}),
         DDim(std::vector<DDim::value_type>{3, 4, 5, 2}),
         true,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 4, 3}),
         DDim(std::vector<DDim::value_type>{5, 6, 4}),
         true,
         true);

    // dim nx1
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 2, 5}),
         DDim(std::vector<DDim::value_type>{5}),
         false,
         false);

    // dim nx2
    test(precision_type,
         DDim(std::vector<DDim::value_type>{1, 2, 2, 3}),
         DDim(std::vector<DDim::value_type>{3, 1}),
         false,
         false);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{1, 2, 2, 3}),
         DDim(std::vector<DDim::value_type>{3, 4}),
         false,
         false);

    // dim nx2 ytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 6, 2}),
         DDim(std::vector<DDim::value_type>{5, 2}),
         false,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3, 5, 2}),
         DDim(std::vector<DDim::value_type>{1, 2}),
         false,
         true);

    // dim nx2 xtranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 6, 2}),
         DDim(std::vector<DDim::value_type>{6, 2}),
         true,
         false);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3, 5, 2}),
         DDim(std::vector<DDim::value_type>{5, 1}),
         true,
         false);

    // dim nx2 xytranspose
    test(precision_type,
         DDim(std::vector<DDim::value_type>{3, 4, 4, 3}),
         DDim(std::vector<DDim::value_type>{2, 4}),
         true,
         true);
    test(precision_type,
         DDim(std::vector<DDim::value_type>{5, 3, 3, 2}),
         DDim(std::vector<DDim::value_type>{1, 3}),
         true,
         true);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(matmul, kOpenCL, kFP16, kImageFolder, image2d);
