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

#include "lite/kernels/cuda/search_aligned_mat_mul_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
void search_aligned_mat_mul_compute_ref(const operators::MatMulParam& param) {
  auto x = param.X;
  auto y = param.Y;
  auto out = param.Out;
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  T alpha = static_cast<T>(param.alpha);
  const auto x_dims = x->dims();
  const auto y_dims = y->dims();
  const auto& x_lod = x->lod();
  const auto& y_lod = y->lod();
  const auto& x_lod_0 = x_lod[0];
  const auto& y_lod_0 = y_lod[0];
  int seq_num = x_lod_0.size() - 1;
  int x_inner_size = x_dims[1];
  int y_inner_size = y_dims[1];
  int x_batch_size = x_lod_0[1];
  int y_batch_size = y_lod_0[1];
  int M = x_transpose ? x_inner_size : x_batch_size;
  int N = y_transpose ? y_batch_size : y_inner_size;
  int X_K = x_transpose ? x_batch_size : x_inner_size;
  int Y_K = y_transpose ? y_inner_size : y_batch_size;
  CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";
  int K = X_K;
  int lda = x_transpose ? M : K;
  int ldb = y_transpose ? K : N;
  int ldc = N;
  int x_stride = x_batch_size * x_inner_size;
  int y_stride = y_batch_size * y_inner_size;
  int out_stride = M * N;
  auto x_data = x->data<T>();
  auto y_data = y->data<T>();
  auto out_data = out->mutable_data<T>();

  for (int seq = 0; seq < seq_num; seq++) {
    auto a = x_data + seq * x_stride;
    auto b = y_data + seq * y_stride;
    auto c = out_data + seq * out_stride;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        auto sum = static_cast<T>(0);
        for (int l = 0; l < K; l++) {
          T av;
          T bv;
          if (x_transpose) {
            av = a[l * lda + i];
          } else {
            av = a[i * lda + l];
          }
          if (y_transpose) {
            bv = b[j * ldb + l];
          } else {
            bv = b[l * ldb + j];
          }
          sum += av * bv;
        }
        c[i * ldc + j] = alpha * sum;
      }
    }
  }
}

TEST(search_aligned_mat_mul_compute, normal) {
  Env<TargetType::kCUDA>::Init();
  for (int seq_num : {1, 2}) {
    for (int x_batch_size : {1, 3}) {
      for (int x_inner_size : {1, 5}) {
        for (int out_inner_size : {1, 4}) {
          for (bool x_transpose : {true, false}) {
            for (bool y_transpose : {true, false}) {
              for (float alpha : {1., 2.}) {
                // infer x_dims and y_dims
                int y_batch_size;
                int y_inner_size;
                int out_batch_size;
                if (x_transpose) {
                  if (y_transpose) {
                    y_batch_size = out_inner_size;
                    y_inner_size = x_batch_size;
                    out_batch_size = x_inner_size;
                  } else {
                    y_batch_size = x_batch_size;
                    y_inner_size = out_inner_size;
                    out_batch_size = x_inner_size;
                  }
                } else {
                  if (y_transpose) {
                    y_batch_size = out_inner_size;
                    y_inner_size = x_inner_size;
                    out_batch_size = x_batch_size;
                  } else {
                    y_batch_size = x_inner_size;
                    y_inner_size = out_inner_size;
                    out_batch_size = x_batch_size;
                  }
                }
                std::vector<uint64_t> x_lod_0(seq_num + 1);
                std::vector<uint64_t> y_lod_0(seq_num + 1);
                std::vector<uint64_t> out_lod_0(seq_num + 1);
                x_lod_0[0] = 0;
                y_lod_0[0] = 0;
                out_lod_0[0] = 0;
                for (int i = 0; i < seq_num; i++) {
                  x_lod_0[i + 1] = x_lod_0[i] + x_batch_size;
                  y_lod_0[i + 1] = y_lod_0[i] + y_batch_size;
                  out_lod_0[i + 1] = out_lod_0[i] + out_batch_size;
                }
                LoD x_lod;
                LoD y_lod;
                LoD out_lod;
                x_lod.push_back(x_lod_0);
                y_lod.push_back(y_lod_0);
                out_lod.push_back(out_lod_0);
                DDim x_dims({static_cast<int64_t>(x_lod_0.back()),
                             static_cast<int64_t>(x_inner_size)});
                DDim y_dims({static_cast<int64_t>(y_lod_0.back()),
                             static_cast<int64_t>(y_inner_size)});
                DDim out_dims({static_cast<int64_t>(out_lod_0.back()),
                               static_cast<int64_t>(out_inner_size)});
                // prepare input&output tensors
                Tensor x_dev, x_host, y_dev, y_host, out_dev, out_host, out_ref;
                x_host.Resize(x_dims);
                y_host.Resize(y_dims);
                out_host.Resize(out_dims);
                x_dev.Resize(x_dims);
                y_dev.Resize(y_dims);
                out_dev.Resize(out_dims);
                out_ref.Resize(out_dims);
                x_host.set_lod(x_lod);
                y_host.set_lod(y_lod);
                out_host.set_lod(out_lod);
                x_dev.set_lod(x_lod);
                y_dev.set_lod(y_lod);
                out_dev.set_lod(out_lod);
                out_ref.set_lod(out_lod);
                auto out_dev_data = out_dev.mutable_data<float>(TARGET(kCUDA));
                auto x_host_data = x_host.mutable_data<float>();
                auto y_host_data = y_host.mutable_data<float>();
                auto out_host_data = out_host.mutable_data<float>();
                auto out_ref_data = out_ref.mutable_data<float>();
                for (int i = 0; i < x_host.dims().production(); i++) {
                  x_host_data[i] = i * 0.125f;
                }
                for (int i = 0; i < y_host.dims().production(); i++) {
                  y_host_data[i] = i * 0.5f;
                }
                x_dev.Assign<float, lite::DDim, TARGET(kCUDA)>(x_host_data,
                                                               x_host.dims());
                y_dev.Assign<float, lite::DDim, TARGET(kCUDA)>(y_host_data,
                                                               y_host.dims());
                // prepare cuda context, initialize param, and run kernel
                operators::MatMulParam param;
                param.X = &x_dev;
                param.Y = &y_dev;
                param.Out = &out_dev;
                param.alpha = alpha;
                param.transpose_X = x_transpose;
                param.transpose_Y = y_transpose;
                std::unique_ptr<KernelContext> ctx(new KernelContext);
                auto& cuda_ctx = ctx->As<CUDAContext>();
                cuda_ctx.InitOnce();
                int dev_id = TargetWrapper<TargetType::kCUDA>::GetCurDevice();
                cuda_ctx.Init(dev_id);
                SearchAlignedMatMulCompute search_aligned_mat_mul;
                search_aligned_mat_mul.SetParam(param);
                search_aligned_mat_mul.SetContext(std::move(ctx));
                search_aligned_mat_mul.Launch();
                cudaDeviceSynchronize();
                CopySync<TARGET(kCUDA)>(
                    out_host_data,
                    out_dev_data,
                    sizeof(float) * out_dev.dims().production(),
                    IoDirection::DtoH);
                // run reference
                param.X = &x_host;
                param.Y = &y_host;
                param.Out = &out_ref;
                search_aligned_mat_mul_compute_ref<float>(param);
                // verify result
                for (int i = 0; i < out_ref.dims().production(); i++) {
                  EXPECT_NEAR(out_host_data[i], out_ref_data[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
