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
#include <random>
#include "lite/core/context.h"
#include "lite/tests/cv/cv_basic.h"
#include "lite/tests/utils/timer.h"
#if 0
// #include "lite/utils/cv/image_convert.h"
#ifdef LITE_WITH_ARM
#include "lite/utils/cv/image_convert.h"
#endif  // LITE_WITH_ARM

// DEFINE_int32(cluster, 3, "cluster id");
// DEFINE_int32(threads, 1, "threads num");
// DEFINE_int32(warmup, 0, "warmup times");
// DEFINE_int32(repeats, 1, "repeats times");
// DEFINE_bool(basic_test, false, "do all tests");
// DEFINE_bool(check_result, true, "check the result");

// DEFINE_int32(srcFormat, 0, "input image format");
// DEFINE_int32(dstFormat, 1, "output image format");
// DEFINE_int32(srch, 112, "input height");
// DEFINE_int32(srcw, 112, "input width");
int FLAGS_srch = 112;
int FLAGS_srcw = 112;
int FLAGS_cluster = 3;
int FLAGS_threads = 1;
int FLAGS_warmup = 0;
int FLAGS_repeats = 1;
bool FLAGS_basic_test = 0;
bool FLAGS_check_result = 1;
int FLAGS_srcFormat = 0;
int FLAGS_dstFormat = 1;

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;

void fill_host_rand(uint8_t* dio, int64_t size) {
  uint seed = 256;
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = rand_r(&seed) % 256;
  }
}

void data_diff_kernel(const uint8_t* src1_truth,
                      const uint8_t* src2,
                      int size,
                      double& max_ratio,   // NOLINT
                      double& max_diff) {  // NOLINT
  const double eps = 1e-6f;
  max_diff = fabs(src1_truth[0] - src2[0]);
  max_ratio = fabs(max_diff) / (std::abs(src1_truth[0]) + eps);
  for (int i = 1; i < size; ++i) {
    double diff = fabs(src1_truth[i] - src2[i]);
    double ratio = fabs(diff) / (std::abs(src1_truth[i]) + eps);
    if (max_ratio < ratio) {
      max_diff = diff;
      max_ratio = ratio;
    }
  }
}

void print_tensor_host_impl(const uint8_t* din, int64_t size, int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%d ", din[i]);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

#ifdef LITE_WITH_ARM
void test_image_convert(const std::vector<int>& thread_num,
                        const std::vector<int>& cluster_id,
                        const int srch,
                        const int srcw,
                        int srcFormat,
                        int dstFormat) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = 3 * srch * srcw / 2;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_host_rand(src, size);

      int out_size = size;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = 3 * srch * srcw / 2;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * srch * srcw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
      }
      // out
      uint8_t* lite_dst = new uint8_t[out_size];
      uint8_t* basic_dst = new uint8_t[out_size];
      if (FLAGS_check_result) {
        image_convert_basic(src,
                            basic_dst,
                            (ImageFormat)srcFormat,
                            (ImageFormat)dstFormat,
                            srcw,
                            srch);
      }
      paddle::lite::utils::cv::ImageConvert img_convert;
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        img_convert.choose(src,
                           lite_dst,
                           (ImageFormat)srcFormat,
                           (ImageFormat)dstFormat,
                           srcw,
                           srch);
      }  /// compute
      lite::test::Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.start();
        img_convert.choose(src,
                           lite_dst,
                           (ImageFormat)srcFormat,
                           (ImageFormat)dstFormat,
                           srcw,
                           srch);
        t0.end();
      }
      double gops = 2.0 * size;
      LOG(INFO) << "image convert: input size:[ " << srch << ", " << srcw
                << "] "
                << ",running time, avg: " << t0.get_average_ms()
                << ", min time: " << t0.get_min_time()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.get_average_ms()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.get_min_time();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        data_diff_kernel(basic_dst, lite_dst, out_size, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 5e-4f) {
            LOG(WARNING) << "basic result";
            print_tensor_host_impl(basic_dst, out_size, srcw);
            LOG(WARNING) << "saber result";
            print_tensor_host_impl(lite_dst, out_size, srcw);
            LOG(WARNING) << "diff result";
            for (int i = 0; i < out_size; i++) {
              int diff = lite_dst[i] - basic_dst[i];
              printf("%d  ", diff);
              if (i % srcw == 0) {
                printf("\n");
              }
            }
            LOG(FATAL) << "test image convert: input: " << size
                       << ", output: " << out_size
                       << ", srcFormat: " << srcFormat
                       << ", dstFormat: " << dstFormat << ", threads: " << th
                       << ", cluster: " << cls << " failed!!\n";
          }
        }
      }
      LOG(INFO) << "test image convert: input: " << size
                << ", output: " << out_size << ", srcFormat: " << srcFormat
                << ", dstFormat: " << dstFormat << ", threads: " << th
                << ", cluster: " << cls << " successed!!\n";
    }
  }
}
#endif

#if 1
TEST(TestImageConvertRand, test_image_convert_rand_size) {
  if (FLAGS_basic_test) {
    for (auto srcFormat : {0, 1, 2, 3, 4, 5, 6}) {
      for (auto dstFormat : {0, 1, 2, 3, 4, 5, 6}) {
        for (auto srch : {2, 4, 8, 32, 64, 112, 224}) {
          for (auto srcw : {4, 8, 32, 64, 70, 110, 112, 224}) {
            if ((srcFormat == ImageFormat::BGR &&
                 dstFormat == ImageFormat::BGRA) ||
                (srcFormat == ImageFormat::RGB &&
                 dstFormat == ImageFormat::RGBA) ||
                (srcFormat == ImageFormat::BGR &&
                 dstFormat == ImageFormat::RGBA) ||
                (srcFormat == ImageFormat::RGB &&
                 dstFormat == ImageFormat::BGRA) ||
                 (srcFormat == ImageFormat::BGRA &&
                 dstFormat == ImageFormat::BGR) ||
                (srcFormat == ImageFormat::RGBA &&
                 dstFormat == ImageFormat::RGB) ||
                (srcFormat == ImageFormat::BGRA &&
                 dstFormat == ImageFormat::RGB) ||
                (srcFormat == ImageFormat::RGBA &&
                 dstFormat == ImageFormat::BGR) ||
                (srcFormat == ImageFormat::BGR &&
                 (dstFormat == ImageFormat::NV12 ||
                  dstFormat == ImageFormat::NV21)) ||
                (srcFormat == ImageFormat::RGB &&
                 (dstFormat == ImageFormat::NV12 ||
                  dstFormat == ImageFormat::NV21)) ||
                  (dstFormat == ImageFormat::GRAY &&
                 (srcFormat == ImageFormat::RGBA ||
                  srcFormat == ImageFormat::BGRA))) {
              continue;
            }
            test_image_convert(
                {1, 2, 4}, {FLAGS_cluster}, srch, srcw, srcFormat, dstFormat);
          }
        }
      }
    }
  }
}
#endif

#if 1
TEST(TestImageConvertCustom, test_image_convert_custom_size) {
  test_image_convert({FLAGS_threads},
                     {FLAGS_cluster},
                     FLAGS_srch,
                     FLAGS_srcw,
                     FLAGS_srcFormat,
                     FLAGS_dstFormat);
}
#endif
#endif
