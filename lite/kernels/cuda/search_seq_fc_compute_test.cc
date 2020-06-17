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

#include "lite/kernels/cuda/search_seq_fc_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
void search_seq_fc_compute_ref(const operators::SearchSeqFcParam& param) {
  auto x = param.x;
  auto w = param.w;
  auto b = param.b;
  auto out = param.out;
  auto out_size = param.out_size;
  const auto x_dims = x->dims();
  const auto w_dims = w->dims();
  const auto& x_lod = x->lod();
  CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
  CHECK(!x_lod.empty()) << "The Input(X) must hold lod info.";
  const auto& x_lod_0 = x_lod[0];
  CHECK_GE(x_lod_0.size(), 2) << "The Input(X)'s lod info is corrupted.";
  CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod_0.back()))
      << "The Input(X)'s lod info mismatches the actual tensor shape.";
  CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
  CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
  CHECK_EQ(w_dims[0], out_size) << "Wrong shape: w_dims[0] != out_size";
  int M = x_dims[0];
  int K = x_dims[1];
  int N = w_dims[0];
  auto x_data = x->data<T>();
  auto w_data = w->data<T>();
  auto out_data = out->mutable_data<T>();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      auto sum = static_cast<T>(0);
      for (int l = 0; l < K; l++) {
        T xv = x_data[i * K + l];
        T wv = w_data[j * K + l];
        sum += xv * wv;
      }
      out_data[i * N + j] = sum;
    }
  }

  if (b != nullptr) {
    auto b_dims = b->dims();
    CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
    CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
    auto b_data = b->data<T>();
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        out_data[i * N + j] += b_data[j];
      }
    }
  }
}

TEST(search_seq_fc_compute, normal) {
  Env<TargetType::kCUDA>::Init();
  for (auto x_lod_0 : {std::vector<uint64_t>({0, 1, 3}),
                       std::vector<uint64_t>({0, 3, 4, 5})}) {
    for (auto feature_size : {2, 9}) {
      for (auto out_size : {3, 5}) {
        for (auto has_bias : {true, false}) {
          // infer x_dims, w_dims, b_dims and out_dims
          DDim x_dims({static_cast<int64_t>(x_lod_0.back()), feature_size});
          DDim w_dims({out_size, feature_size});
          DDim b_dims({has_bias ? out_size : 0});
          DDim out_dims({static_cast<int64_t>(x_lod_0.back()), out_size});
          LoD x_lod;
          x_lod.push_back(x_lod_0);
          LoD out_lod;
          out_lod.push_back(x_lod_0);
          // prepare input&output tensors
          Tensor x_dev, x_host, w_dev, w_host, b_dev, b_host, out_dev, out_host,
              out_ref;
          x_host.Resize(x_dims);
          w_host.Resize(w_dims);
          b_host.Resize(b_dims);
          out_host.Resize(out_dims);
          x_dev.Resize(x_dims);
          w_dev.Resize(w_dims);
          b_dev.Resize(b_dims);
          out_dev.Resize(out_dims);
          out_ref.Resize(out_dims);
          x_host.set_lod(x_lod);
          out_host.set_lod(out_lod);
          x_dev.set_lod(x_lod);
          out_dev.set_lod(out_lod);
          out_ref.set_lod(out_lod);
          auto out_dev_data = out_dev.mutable_data<float>(TARGET(kCUDA));
          auto x_host_data = x_host.mutable_data<float>();
          auto w_host_data = w_host.mutable_data<float>();
          auto out_host_data = out_host.mutable_data<float>();
          auto out_ref_data = out_ref.mutable_data<float>();
          for (int i = 0; i < x_host.dims().production(); i++) {
            x_host_data[i] = i * 0.125f;
          }
          for (int i = 0; i < w_host.dims().production(); i++) {
            w_host_data[i] = i * 0.5f;
          }
          x_dev.Assign<float, lite::DDim, TARGET(kCUDA)>(x_host_data,
                                                         x_host.dims());
          w_dev.Assign<float, lite::DDim, TARGET(kCUDA)>(w_host_data,
                                                         w_host.dims());
          // prepare cuda context, initialize param, and run kernel
          operators::SearchSeqFcParam param;
          param.x = &x_dev;
          param.w = &w_dev;
          param.out = &out_dev;
          param.out_size = out_size;
          if (has_bias) {
            auto b_host_data = b_host.mutable_data<float>();
            for (int i = 0; i < b_host.dims().production(); i++) {
              b_host_data[i] = i * 0.5f;
            }
            b_dev.Assign<float, lite::DDim, TARGET(kCUDA)>(b_host_data,
                                                           b_host.dims());
            param.b = &b_dev;
          }
          std::unique_ptr<KernelContext> ctx(new KernelContext);
          auto& cuda_ctx = ctx->As<CUDAContext>();
          cuda_ctx.InitOnce();
          int dev_id = TargetWrapper<TargetType::kCUDA>::GetCurDevice();
          cuda_ctx.Init(dev_id);
          SearchSeqFcCompute search_seq_fc;
          search_seq_fc.SetParam(param);
          search_seq_fc.SetContext(std::move(ctx));
          search_seq_fc.Launch();
          cudaDeviceSynchronize();
          CopySync<TARGET(kCUDA)>(out_host_data,
                                  out_dev_data,
                                  sizeof(float) * out_dev.dims().production(),
                                  IoDirection::DtoH);
          // run reference
          param.x = &x_host;
          param.w = &w_host;
          param.out = &out_ref;
          if (has_bias) {
            param.b = &b_host;
          }
          search_seq_fc_compute_ref<float>(param);
          // verify result
          for (int i = 0; i < out_ref.dims().production(); i++) {
            EXPECT_NEAR(out_host_data[i], out_ref_data[i], 1e-5);
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
