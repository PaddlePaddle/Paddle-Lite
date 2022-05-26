// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/tests/utils/fill_data.h"

#define FP16_RELATIVE_DIFF (5e-2)
#define FP16_ABS_DIFF (5e-2)
#define FP32_RELATIVE_DIFF (1e-3)
#define FP32_ABS_DIFF (5e-4)

namespace paddle {
namespace lite {
float lrn_square(const float* din,
                 int c,
                 int offset,
                 int channel,
                 int height,
                 int width,
                 int local_size) {
  int pre_pad = (local_size - 1) / 2;
  float sum = 0.f;
  int start = c - pre_pad;
  int end = c + pre_pad;
  start = start < 0 ? 0 : start;
  end = end < channel - 1 ? end : channel - 1;
  for (int i = start; i <= end; i++) {
    sum += din[i * height * width] * din[i * height * width];
  }
  return sum;
}
void lrn_ref(const float* din,
             const DDim& in_dims,
             float* output,
             int local_size,
             float k,
             float alpha,
             float beta,
             std::string norm_region) {
  int num = in_dims[0];
  int channel = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];

  if (norm_region == "AcrossChannels") {
    for (int b = 0; b < num; b++) {
      const float* din_batch = din + b * channel * height * width;
      float* dout_batch = output + b * channel * height * width;
      int offset_num = b * channel * height * width;
      for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int offset_within_channel = h * width + w;
            int dst_id = c * height * width + offset_within_channel;
            float square = lrn_square(din_batch,
                                      c,
                                      offset_within_channel,
                                      channel,
                                      height,
                                      width,
                                      local_size);
            dout_batch[dst_id] =
                din_batch[dst_id] * pow(k + alpha * square, -beta);
          }
        }
      }
    }
  }
}
// #define LRN_FP16_LOOP_TEST
// #define LRN_FP16_PRINT_RESULT
void test_precision(const lite_api::CLPrecisionType p) {
#ifdef LRN_FP16_LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3, 8, 23, 32}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
          for (auto num : {3, 5, 9}) {
            for (auto k : {1.0, 1.5}) {
              for (auto alpha : {1e-4}) {
                for (auto beta : {0.5, 0.75}) {
                  for (auto norm_region : {"AcrossChannels"}) {
#else
  const int n = 1;
  const int c = 5;
  const int h = 2;
  const int w = 4;
  const int num = 5;
  const float k = 1.0;
  const float alpha = 1e-4;
  const float beta = 0.75;
  const std::string norm_region = "AcrossChannels";
#endif  // GRID_FP16_LOOP_TEST

                    LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " "
                              << c << " " << h << " " << w << " ========";
                    LOG(INFO) << "LRN parameters: ";
                    LOG(INFO) << "num: " << num << ", k: " << k
                              << ", alpha: " << alpha << ", beta: " << beta
                              << ", norm_region: " << norm_region;
                    CLRuntime::Global()->set_precision(p);
                    const bool fp16_flag =
                        (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
                    auto kernels = KernelRegistry::Global().Create(
                        "lrn",
                        TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault));
                    ASSERT_FALSE(kernels.empty());
                    auto kernel = std::move(kernels.front());
                    LOG(INFO) << "get kernel:" << kernel->doc();

                    lite::Tensor x, out;
                    operators::LrnParam param;
                    param.X = &x;
                    param.Out = &out;
                    param.n = num;
                    param.k = k;
                    param.alpha = alpha;
                    param.beta = beta;
                    param.norm_region = norm_region;

                    std::unique_ptr<KernelContext> context(new KernelContext);
                    context->As<OpenCLContext>().InitOnce();

                    kernel->SetParam(param);
                    std::unique_ptr<KernelContext> lrn_context(
                        new KernelContext);
                    context->As<OpenCLContext>().CopySharedTo(
                        &(lrn_context->As<OpenCLContext>()));
                    kernel->SetContext(std::move(lrn_context));

                    const DDim in_dim =
                        DDim(std::vector<DDim::value_type>{n, c, h, w});
                    const DDim out_dim =
                        DDim(std::vector<DDim::value_type>{n, c, h, w});
                    x.Resize(in_dim);
                    out.Resize(out_dim);

                    std::vector<float> input_v(in_dim.production());
                    std::vector<float> output_v(out_dim.production());
                    fill_data_rand(
                        input_v.data(), -1.f, 1.f, in_dim.production());

                    LOG(INFO) << "prepare input";
                    CLImageConverterDefault* default_converter =
                        new CLImageConverterDefault();
                    DDim x_image_shape =
                        default_converter->InitImageDimInfoWith(in_dim);
                    LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
                              << x_image_shape[1];
                    const size_t dtype_size =
                        fp16_flag ? sizeof(half_t) : sizeof(float);
                    std::vector<char> x_image_data(x_image_shape.production() *
                                                   4 * dtype_size);  // 4 : RGBA
                    default_converter->NCHWToImage(
                        input_v.data(), x_image_data.data(), in_dim);
                    MUTABLE_DATA_GPU(&x,
                                     x_image_shape[0],
                                     x_image_shape[1],
                                     x_image_data.data());

                    DDim out_image_shape =
                        default_converter->InitImageDimInfoWith(out_dim);
                    LOG(INFO) << "out_image_shape = " << out_image_shape[0]
                              << " " << out_image_shape[1];
                    auto* out_image = MUTABLE_DATA_GPU(
                        &out, out_image_shape[0], out_image_shape[1], nullptr);

                    kernel->Launch();

                    CLRuntime::Global()->command_queue().finish();

                    std::vector<float> out_ref(out_dim.production());
                    auto* out_ref_data = out_ref.data();
                    lrn_ref(input_v.data(),
                            in_dim,
                            out_ref_data,
                            num,
                            k,
                            alpha,
                            beta,
                            norm_region);

                    const size_t cl_image2d_row_pitch{0};
                    const size_t cl_image2d_slice_pitch{0};

                    std::vector<char> out_image_data(
                        out_image_shape.production() * 4 *
                        dtype_size);  // 4 : RGBA
                    TargetWrapperCL::ImgcpySync(out_image_data.data(),
                                                out_image,
                                                out_image_shape[0],
                                                out_image_shape[1],
                                                cl_image2d_row_pitch,
                                                cl_image2d_slice_pitch,
                                                IoDirection::DtoH);
                    default_converter->ImageToNCHW(out_image_data.data(),
                                                   output_v.data(),
                                                   out_image_shape,
                                                   out_dim);
// result
#ifdef LRN_FP16_PRINT_RESULT
                    LOG(INFO)
                        << "---- print kernel result (input -> output) ----";
                    for (int eidx = 0; eidx < in_dim.production(); ++eidx) {
                      std::cout << input_v[eidx] << " -> " << out_data[eidx]
                                << std::endl;
                    }
#endif  // LRN_FP16_PRINT_RESULT
                    uint32_t diff_cnt = 0;
                    auto relative_diff_thres =
                        fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
                    auto abs_diff_thres =
                        fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
                    for (int i = 0; i < out_dim.production(); i++) {
                      auto abs_diff = abs(output_v[i] - out_ref_data[i]);
                      auto relative_diff =
                          COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                      EXPECT_FALSE(relative_diff > relative_diff_thres &&
                                   abs_diff > abs_diff_thres);
                      if ((relative_diff > relative_diff_thres) &&
                          (abs_diff > abs_diff_thres)) {
                        LOG(WARNING) << i << ": \t out_ins: " << output_v[i]
                                     << "\t out_ref: " << out_ref_data[i];
                        diff_cnt++;
                      }
                    }
                    if (diff_cnt != 0) {
                      LOG(FATAL) << "Err num " << diff_cnt << "/"
                                 << out_dim.production();
                    }
#ifdef LRN_FP16_LOOP_TEST
                  }  // norm_region
                }    // beta
              }      // alpha
            }        // k
          }          // num
        }            // w
      }              // h
    }                // c
  }                  // n
#else
// nothing to do.
#endif
}

TEST(lrn, compute_basic) {
  for (auto p : {lite_api::CLPrecisionType::CL_PRECISION_FP32,
                 lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    test_precision(p);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(lrn, kOpenCL, kFP16, kImageDefault, ImageDefault);
