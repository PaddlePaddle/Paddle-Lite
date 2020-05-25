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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-3)
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");

using paddle::lite::profile::Timer;

namespace paddle {
namespace lite {
void instance_norm_ref(Tensor* x,
                       Tensor* y,
                       Tensor* scale,
                       Tensor* bias,
                       Tensor* saved_mean,
                       Tensor* saved_variance,
                       float epsilon) {
  auto x_data = x->data<float>();
  auto scale_data = scale->data<float>();
  auto bias_data = bias->data<float>();
  auto y_data = y->mutable_data<float>();
  auto saved_mean_data = saved_mean->mutable_data<float>();
  auto saved_variance_data = saved_variance->mutable_data<float>();
  int n = x->dims()[0];
  int c = x->dims()[1];
  int spatial_size = x->dims()[2] * x->dims()[3];

  // compute mean
  for (int i = 0; i < n * c; ++i) {
    const float* x_ptr = x_data + i * spatial_size;
    float sum = 0.f;
    for (int j = 0; j < spatial_size; ++j) {
      sum += x_ptr[j];
    }
    saved_mean_data[i] = sum / spatial_size;
  }
  // compute variance
  for (int i = 0; i < n * c; ++i) {
    const float* x_ptr = x_data + i * spatial_size;
    float sum = 0.f;
    for (int j = 0; j < spatial_size; ++j) {
      sum += (x_ptr[j] - saved_mean_data[i]) * (x_ptr[j] - saved_mean_data[i]);
    }
    saved_variance_data[i] = 1.f / sqrtf(sum / spatial_size + epsilon);
  }
  // compute out
  for (int i = 0; i < n * c; ++i) {
    const float* x_ptr = x_data + i * spatial_size;
    float* y_ptr = y_data + i * spatial_size;
    float scale_val = scale_data[i % c];
    float bias_val = bias_data[i % c];
    for (int j = 0; j < spatial_size; ++j) {
      y_ptr[j] =
          scale_val * (x_ptr[j] - saved_mean_data[i]) * saved_variance_data[i] +
          bias_val;
    }
  }
}

// #define INSTANCE_NORM_FP16_LOOP_TEST
// #define INSTANCE_NORM_FP16_PRINT_RESULT
TEST(instance_norm_image2d, compute) {
#ifdef INSTANCE_NORM_FP16_LOOP_TEST
  for (auto n : {1, 3}) {
    for (auto c : {1, 3, 8, 32, 65}) {
      for (auto h : {4, 20, 64, 112, 224}) {
        for (auto w : {2, 20, 64, 112, 224}) {
#else
  const int n = 1;
  const int c = 32;
  const int h = 224;
  const int w = 224;
#endif  // INSTANCE_NORM_FP16_LOOP_TEST

          LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                    << h << " " << w << " ========";

          auto kernels =
              KernelRegistry::Global().Create("instance_norm",
                                              TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(kernels.empty());
          auto kernel = std::move(kernels.front());
          LOG(INFO) << "get kernel:" << kernel->doc();

          lite::Tensor x, out, out_ref, scale, bias, saved_mean, saved_variance;
          operators::InstanceNormParam param;
          param.x = &x;
          param.out = &out;
          param.scale = &scale;
          param.bias = &bias;
          param.saved_mean = &saved_mean;
          param.saved_variance = &saved_variance;
          param.epsilon = 1e-5;
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          kernel->SetParam(param);
          std::unique_ptr<KernelContext> instance_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(instance_context->As<OpenCLContext>()));
          kernel->SetContext(std::move(instance_context));

          const DDim in_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(in_dim);
          out.Resize(in_dim);
          out_ref.Resize(in_dim);
          scale.Resize({c});
          bias.Resize({c});
          saved_mean.Resize({n * c});
          saved_variance.Resize({n * c});
          auto* x_data = x.mutable_data<float>();
          auto* scale_data = scale.mutable_data<float>();
          auto* bias_data = bias.mutable_data<float>();
          auto* saved_mean_data = saved_mean.mutable_data<float>();
          auto* saved_variance_data = saved_variance.mutable_data<float>();
          std::default_random_engine engine;
          std::uniform_real_distribution<float> dist(-1, 1);
          int sum = n * c * h * w;
          for (int i = 0; i < sum; ++i) {
            x_data[i] = dist(engine);
          }
          for (int i = 0; i < c; ++i) {
            scale_data[i] = dist(engine);
            bias_data[i] = dist(engine);
          }
          //! run reference instance norm
          instance_norm_ref(
              &x, &out_ref, &scale, &bias, &saved_mean, &saved_variance, 1e-5);
          LOG(INFO) << "prepare input";
          CLImageConverterDefault* default_converter =
              new CLImageConverterDefault();
          DDim x_image_shape = default_converter->InitImageDimInfoWith(in_dim);
          LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
                    << x_image_shape[1];
          std::vector<half_t> x_image_data(x_image_shape.production() *
                                           4);  // 4 : RGBA
          default_converter->NCHWToImage(x_data, x_image_data.data(), in_dim);
          auto* x_image = x.mutable_data<half_t, cl::Image2D>(
              x_image_shape[0], x_image_shape[1], x_image_data.data());

          auto* out_image = out.mutable_data<half_t, cl::Image2D>(
              x_image_shape[0], x_image_shape[1]);

          //! warm up
          for (int i = 0; i < FLAGS_warmup; ++i) {
            kernel->Launch();
          }
          context->As<OpenCLContext>().cl_context()->GetCommandQueue().finish();
          //! compute
          Timer t0;
          t0.Start();
          for (int i = 0; i < FLAGS_repeats; ++i) {
            kernel->Launch();
          }
          context->As<OpenCLContext>().cl_context()->GetCommandQueue().finish();
          t0.Stop();
          double gops = 6 * sum;
          LOG(INFO) << "avg time: " << t0.LapTimes().Avg() / FLAGS_repeats
                    << " ms, "
                    << "avg GOPs: "
                    << 1e-6 * gops * FLAGS_repeats / t0.LapTimes().Avg()
                    << " GOPs";
          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};
          half_t* out_image_data = new half_t[x_image_shape.production() * 4];
          TargetWrapperCL::ImgcpySync(out_image_data,
                                      out_image,
                                      x_image_shape[0],
                                      x_image_shape[1],
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          float* out_data = new float[x_image_shape.production() * 4];
          default_converter->ImageToNCHW(
              out_image_data, out_data, x_image_shape, in_dim);
// result
#ifdef INSTANCE_NORM_FP16_PRINT_RESULT
          LOG(INFO) << "---- print kernel result (input -> output) ----";
          for (int eidx = 0; eidx < in_dim.production(); ++eidx) {
            std::cout << x_data[eidx] << " -> " << out_data[eidx] << std::endl;
          }
#endif  // INSTANCE_NORM_FP16_PRINT_RESULT
          auto* out_ref_data = out_ref.data<float>();
          for (int i = 0; i < in_dim.production(); i++) {
            auto abs_diff = abs(out_data[i] - out_ref_data[i]);
            auto relative_diff =
                COMPUTE_RELATIVE_DIFF(out_data[i], out_ref_data[i]);
            EXPECT_EQ(
                (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
                true);
            if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
              LOG(ERROR) << "error idx:" << i << ", in_data[" << i
                         << "]: " << x_data[i] << ", out_data[" << i
                         << "]: " << out_data[i] << ", out_ref[" << i
                         << "]: " << out_ref_data[i]
                         << ", abs_diff: " << abs_diff
                         << ", relative_diff: " << relative_diff
                         << ", FP16_MAX_DIFF: " << FP16_MAX_DIFF;
            }
          }
          delete[] out_data;
          delete[] out_image_data;
#ifdef INSTANCE_NORM_FP16_LOOP_TEST
        }  // w
      }    // h
    }      // c
  }        // n
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(instance_norm, kOpenCL, kFP16, kImageDefault, ImageDefault);
