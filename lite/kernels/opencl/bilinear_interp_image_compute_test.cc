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
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {
void bilinear_interp_ref(const float* din,
                         const DDim& x_dims,
                         float* dout,
                         const DDim& out_dims,
                         bool align_corners,
                         int align_mode) {
  int batch_size = x_dims[0];
  int channel_size = x_dims[1];
  auto in_h = x_dims[2];
  auto in_w = x_dims[3];

  int out_h = out_dims[2];
  int out_w = out_dims[3];

  // copy from x if no change
  if (in_h == out_h && in_w == out_w) {
    memcpy(dout, din, sizeof(float) * x_dims.production());
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  // naive bilinear interpolation
  bool align_flag = (align_mode == 0 && !align_corners);

  for (int n = 0; n < batch_size; n++) {
    float* dout_data = dout + n * channel_size * out_h * out_w;
    const float* din_data = din + n * channel_size * in_h * in_w;
    for (int c = 0; c < channel_size; c++) {
      float* dout_data_c = dout_data + c * out_h * out_w;
      const float* din_data_c = din_data + c * in_h * in_w;
      for (int h = 0; h < out_h; h++) {
        float center_h = align_flag ? (ratio_h * (h + 0.5) - 0.5) : ratio_h * h;
        int floor_h = static_cast<int>(center_h);
        int ceil_h = floor_h + 1;
        floor_h = floor_h > 0 ? floor_h : 0;
        ceil_h = ceil_h > in_h - 1 ? in_h - 1 : ceil_h;
        float hs = center_h - floor_h;
        float he = 1.0 - hs;
        for (int w = 0; w < out_w; w++) {
          float center_w =
              align_flag ? (ratio_w * (w + 0.5) - 0.5) : ratio_w * w;
          int floor_w = static_cast<int>(center_w);
          int ceil_w = floor_w + 1;
          floor_w = floor_w > 0 ? floor_w : 0;
          ceil_w = ceil_w > in_w - 1 ? in_w - 1 : ceil_w;
          float ws = center_w - floor_w;
          float we = 1.0 - ws;
          float left_up = din_data_c[ceil_h * in_w + floor_w] * we * hs;
          float left_down = din_data_c[floor_h * in_w + floor_w] * we * he;
          float right_up = din_data_c[ceil_h * in_w + ceil_w] * ws * hs;
          float right_down = din_data_c[floor_h * in_w + ceil_w] * ws * he;
          dout_data_c[h * out_w + w] =
              left_up + left_down + right_up + right_down;
        }
      }
    }
  }
}
// #define BILINEAR_FP16_LOOP_TEST
// #define BILINEAR_FP16_PRINT_RESULT
TEST(bilinear_interp_image2d, compute) {
#ifdef BILINEAR_FP16_LOOP_TEST
  for (auto n : {1, 3}) {
    for (auto c : {1, 3, 8, 23, 32}) {
      for (auto h : {2, 20, 64, 112}) {
        for (auto w : {2, 20, 64, 112}) {
          for (auto out_h : {4, 32, 96, 224}) {
            for (auto out_w : {4, 32, 96, 224}) {
              for (auto align_corners : {true, false}) {
                for (auto align_mode : {0, 1}) {
#else
  const int n = 1;
  const int c = 1;
  const int h = 2;
  const int w = 2;
  const int out_h = 4;
  const int out_w = 4;
  const bool align_corners = true;
  const int align_mode = 0;
#endif  // BILINEAR_FP16_LOOP_TEST

                  LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c
                            << " " << h << " " << w << " ========";
                  LOG(INFO) << "======== parameters: out_h = " << out_h
                            << ", out_w = " << out_w;
                  LOG(INFO) << "align_corners: " << align_corners
                            << ", align_mode: " << align_mode;

                  auto kernels = KernelRegistry::Global().Create(
                      "bilinear_interp",
                      TARGET(kOpenCL),
                      PRECISION(kFP16),
                      DATALAYOUT(kImageDefault));
                  ASSERT_FALSE(kernels.empty());
                  auto kernel = std::move(kernels.front());
                  LOG(INFO) << "get kernel:" << kernel->doc();

                  lite::Tensor x, out;
                  operators::InterpolateParam param;
                  param.X = &x;
                  param.Out = &out;
                  param.align_corners = align_corners;
                  param.align_mode = align_mode;

                  std::unique_ptr<KernelContext> context(new KernelContext);
                  context->As<OpenCLContext>().InitOnce();

                  kernel->SetParam(param);
                  std::unique_ptr<KernelContext> bilinear_context(
                      new KernelContext);
                  context->As<OpenCLContext>().CopySharedTo(
                      &(bilinear_context->As<OpenCLContext>()));
                  kernel->SetContext(std::move(bilinear_context));

                  const DDim in_dim =
                      DDim(std::vector<DDim::value_type>{n, c, h, w});
                  const DDim out_dim =
                      DDim(std::vector<DDim::value_type>{n, c, out_h, out_w});
                  x.Resize(in_dim);
                  out.Resize(out_dim);

                  std::default_random_engine engine;
                  std::uniform_real_distribution<float> dist(-1, 1);
                  int sum = n * c * h * w;
                  std::vector<float> input_v(sum);
                  for (auto& i : input_v) {
                    i = dist(engine);
                  }

                  LOG(INFO) << "prepare input";
                  CLImageConverterDefault* default_converter =
                      new CLImageConverterDefault();
                  DDim x_image_shape =
                      default_converter->InitImageDimInfoWith(in_dim);
                  LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
                            << x_image_shape[1];
                  std::vector<half_t> x_image_data(x_image_shape.production() *
                                                   4);  // 4 : RGBA
                  default_converter->NCHWToImage(
                      input_v.data(), x_image_data.data(), in_dim);
                  auto* x_image = x.mutable_data<half_t, cl::Image2D>(
                      x_image_shape[0], x_image_shape[1], x_image_data.data());

                  DDim out_image_shape =
                      default_converter->InitImageDimInfoWith(out_dim);
                  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
                            << out_image_shape[1];
                  auto* out_image = out.mutable_data<half_t, cl::Image2D>(
                      out_image_shape[0], out_image_shape[1]);

                  kernel->Launch();
                  CLRuntime::Global()->command_queue().finish();

                  std::unique_ptr<float[]> out_ref(
                      new float[out_dim.production()]);
                  bilinear_interp_ref(input_v.data(),
                                      in_dim,
                                      out_ref.get(),
                                      out_dim,
                                      align_corners,
                                      align_mode);

                  const size_t cl_image2d_row_pitch{0};
                  const size_t cl_image2d_slice_pitch{0};
                  half_t* out_image_data =
                      new half_t[40000];  // out_image_shape.production() * 4
                  TargetWrapperCL::ImgcpySync(out_image_data,
                                              out_image,
                                              out_image_shape[0],
                                              out_image_shape[1],
                                              cl_image2d_row_pitch,
                                              cl_image2d_slice_pitch,
                                              IoDirection::DtoH);
                  float* out_data = new float[out_image_shape.production() * 4];
                  default_converter->ImageToNCHW(
                      out_image_data, out_data, out_image_shape, out_dim);
// result
#ifdef BILINEAR_FP16_PRINT_RESULT
                  LOG(INFO)
                      << "---- print kernel result (input -> output) ----";
                  for (int eidx = 0; eidx < in_dim.production(); ++eidx) {
                    std::cout << input_v[eidx] << " -> " << out_data[eidx]
                              << std::endl;
                  }
#endif  // BILINEAR_FP16_PRINT_RESULT
                  for (int i = 0; i < out_dim.production(); i++) {
                    auto abs_diff = abs(out_data[i] - out_ref[i]);
                    auto relative_diff =
                        COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
                    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) ||
                                  (abs_diff <= FP16_MAX_DIFF),
                              true);
                    if ((relative_diff > FP16_MAX_DIFF) &&
                        (abs_diff > FP16_MAX_DIFF)) {
                      LOG(ERROR) << "error idx:" << i << ", in_data[" << i
                                 << "]: " << input_v[i] << ", out_data[" << i
                                 << "]: " << out_data[i] << ", out_ref[" << i
                                 << "]: " << out_ref[i]
                                 << ", abs_diff: " << abs_diff
                                 << ", relative_diff: " << relative_diff
                                 << ", FP16_MAX_DIFF: " << FP16_MAX_DIFF;
                    }
                  }
#ifdef BILINEAR_FP16_LOOP_TEST
                }  // mode
              }    // corners
            }      // out_w
          }        // out_h
        }          // w
      }            // h
    }              // c
  }                // n
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(bilinear_interp, kOpenCL, kFP16, kImageDefault, ImageDefault);
