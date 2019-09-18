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

namespace paddle {
namespace lite {

std::vector<size_t>& InitImageDimInfoWith(const DDim& tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;
  std::vector<size_t> image_shape;
  image_shape.push_back(width);
  image_shape.push_back(height);
  return image_shape;
}

void NCHWToImage(float* nchw, float* image, const DDim& tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t width = in_image_dim[0];
  size_t w_block = width / W;

  float* p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * width * H + h * width + (c / 4) * W + w) * 4 +
            // (c % 4);
            image[i2] = *p;
            i2 += 4;
            p++;
          } else {
            image[i2] = 0.0;
            i2 += 4;
          }
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

void ImageToNCHW(float* image,
                 float* tensor,
                 const DDim& image_dim,
                 const DDim& tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  size_t width = image_dim[0];
  float* p = tensor;

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          *p = image[i2];
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

// #define PRINT_RESULT
#define LOOP_TEST
TEST(fc, compute) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

#ifdef LOOP_TEST
  for (int m = 1; m < 213; m += 71) {
    for (int k = 1; k < 123; k += 31) {
      for (int n = 1; n < 123; n += 121) {
#else
#if 0
  const int m = 1;
  const int k = 1024;
  const int n = 1000;
#else
  const int m = 2;
  const int k = 3;
  const int n = 1;
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

        auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto* w_data = w.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto* bias_data = bias.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

        std::default_random_engine engine;
        std::uniform_real_distribution<float> dist(-5, 5);
        auto* mapped_x = static_cast<float*>(TargetWrapperCL::Map(
            x_data, 0, sizeof(float) * x_dim.production()));
        for (int i = 0; i < x_dim.production(); ++i) {
          mapped_x[i] = static_cast<int>(dist(engine));
        }
        auto* mapped_w = static_cast<float*>(TargetWrapperCL::Map(
            w_data, 0, sizeof(float) * w_dim.production()));
        for (int i = 0; i < w_dim.production(); ++i) {
          mapped_w[i] = static_cast<int>((dist(engine)));
        }
        auto* mapped_bias = static_cast<float*>(TargetWrapperCL::Map(
            bias_data, 0, sizeof(float) * bias_dim.production()));
        for (int i = 0; i < bias_dim.production(); ++i) {
          mapped_bias[i] = static_cast<int>(/*(dist(engine))*/ 1);
        }

        // run opencl kernel
        kernel->Launch();

        auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
        auto* out_ptr = param.output->data<float, cl::Buffer>();
        auto it = wait_list->find(out_ptr);
        if (it != wait_list->end()) {
          VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
          auto& event = *(it->second);
          event.wait();
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

        // run cpu ref
        auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
        gemm_bias<float>(
            mapped_x, m, k, mapped_w, k, n, mapped_bias, out_ref_data);

        auto* out_data = out.mutable_data<float, cl::Buffer>();
        auto* mapped_out = static_cast<float*>(TargetWrapperCL::Map(
            out_data, 0, sizeof(float) * out_dim.production()));

#ifdef PRINT_RESULT
        PrintData("mapped_x", static_cast<float*>(mapped_x), m, k);
        PrintData("mapped_w", static_cast<float*>(mapped_w), k, n);
        PrintData("mapped_bias", static_cast<float*>(mapped_bias), 1, n);
        PrintData("out_ref_data", static_cast<float*>(out_ref_data), m, n);
        PrintData("mapped_out", static_cast<float*>(mapped_out), m, n);
#endif

        for (int i = 0; i < out_dim.production(); i++) {
          EXPECT_NEAR(mapped_out[i], out_ref_data[i], 1e-6);
        }

        TargetWrapperCL::Unmap(x_data, mapped_x);
        TargetWrapperCL::Unmap(w_data, mapped_w);
        TargetWrapperCL::Unmap(bias_data, mapped_bias);
        TargetWrapperCL::Unmap(out_data, mapped_out);
#ifdef LOOP_TEST
      }  // n
    }    // k
  }      // m
#endif
}

}  // namespace lite
}  // namespace paddle

// USE_LITE_KERNEL(fc, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kOpenCL, kFloat, kNCHW, image2d);
