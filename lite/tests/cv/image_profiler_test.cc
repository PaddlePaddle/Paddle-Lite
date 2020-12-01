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
#include <math.h>
#include <random>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/tests/cv/anakin/cv_utils.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/utils/cv/paddle_image_preprocess.h"
#include "time.h"  // NOLINT
DEFINE_int32(cluster, 3, "cluster id");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 10, "repeats times");
DEFINE_bool(basic_test, true, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(srcFormat, 12, "input image format NV12");
DEFINE_int32(dstFormat, 3, "output image format BGR");
DEFINE_int32(srch, 1920, "input height");
DEFINE_int32(srcw, 1080, "input width");
DEFINE_int32(dsth, 960, "output height");
DEFINE_int32(dstw, 540, "output width");
DEFINE_int32(angle, 90, "rotate angel");
DEFINE_int32(flip_num, 0, "flip x");
DEFINE_int32(layout, 1, "layout nchw");

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite_api::DataLayoutType LayoutType;
typedef paddle::lite::utils::cv::TransParam TransParam;
typedef paddle::lite::utils::cv::ImagePreprocess ImagePreprocess;
typedef paddle::lite_api::Tensor Tensor_api;
typedef paddle::lite::Tensor Tensor;

using paddle::lite::profile::Timer;

void fill_tensor_host_rand(uint8_t* dio, int64_t size) {
  uint seed = 256;
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = rand_r(&seed) % 256;  // -128;
  }
}

void print_int8(uint8_t* ptr, int size, int width) {
  for (int i = 0; i < size; i++) {
    printf("%d  ", *ptr++);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

void print_int(int* ptr, int size, int width) {
  int j = 0;
  for (int i = 0; i < size; i++) {
    printf("%d  ", *ptr++);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

void print_fp32(const float* ptr, int size, int width) {
  int j = 0;
  for (int i = 0; i < size; i++) {
    printf("%f  ", *ptr++);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}
#ifdef LITE_WITH_ARM
void test_convert(const std::vector<int>& cluster_id,
                  const std::vector<int>& thread_num,
                  int srcw,
                  int srch,
                  int dstw,
                  int dsth,
                  ImageFormat srcFormat,
                  ImageFormat dstFormat,
                  float rotate,
                  FlipParam flip,
                  LayoutType layout,
                  int test_iter = 10) {
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;
      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_tensor_host_rand(src, size);

      int out_size = srch * srcw;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = ceil(1.5 * srch) * srcw;
      } else if (dstFormat == ImageFormat::BGR ||
                 dstFormat == ImageFormat::RGB) {
        out_size = 3 * srch * srcw;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * srch * srcw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
      }
      uint8_t* basic_dst = new uint8_t[out_size];
      uint8_t* lite_dst = new uint8_t[out_size];
      Timer t_basic, t_lite;
      LOG(INFO) << "basic Convert compute";
      for (int i = 0; i < test_iter; i++) {
        t_basic.Start();
        image_basic_convert(src,
                            basic_dst,
                            (ImageFormat)srcFormat,
                            (ImageFormat)dstFormat,
                            srcw,
                            srch,
                            out_size);
        t_basic.Stop();
      }
      LOG(INFO) << "image baisc Convert avg time : " << t_basic.LapTimes().Avg()
                << ", min time: " << t_basic.LapTimes().Min()
                << ", max time: " << t_basic.LapTimes().Max();

      LOG(INFO) << "lite Convert compute";
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = srch;
      tparam.ow = srcw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        t_lite.Start();
        image_preprocess.image_convert(src, lite_dst);
        t_lite.Stop();
      }
      LOG(INFO) << "image Convert avg time : " << t_lite.LapTimes().Avg()
                << ", min time: " << t_lite.LapTimes().Min()
                << ", max time: " << t_lite.LapTimes().Max();
      LOG(INFO) << "basic Convert compute";

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      if (FLAGS_check_result) {
        LOG(INFO) << "diff, image convert size: " << out_size;
        uint8_t* diff_v = new uint8_t[out_size];
        for (int i = 0; i < out_size; i++) {
          uint8_t a = lite_dst[i];
          uint8_t b = basic_dst[i];
          uint8_t diff1 = a - b;
          uint8_t diff = diff1 > 0 ? diff1 : -diff1;
          diff_v[i] = diff;
          if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = size / srch;
          printf("din: \n");
          print_int8(src, size, width);
          width = out_size / srch;
          printf("saber result: \n");
          print_int8(lite_dst, out_size, width);
          printf("basic result: \n");
          print_int8(basic_dst, out_size, width);
          printf("diff result: \n");
          print_int8(diff_v, out_size, width);
        }
        delete[] diff_v;
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      LOG(INFO) << "image convert end";
    }
  }
}

void test_resize(const std::vector<int>& cluster_id,
                 const std::vector<int>& thread_num,
                 int srcw,
                 int srch,
                 int dstw,
                 int dsth,
                 ImageFormat srcFormat,
                 ImageFormat dstFormat,
                 float rotate,
                 FlipParam flip,
                 LayoutType layout,
                 int test_iter = 10) {
  test_iter = 1;
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_tensor_host_rand(src, size);

      int out_size = dsth * dstw;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = ceil(1.5 * dsth) * dstw;
      } else if (dstFormat == ImageFormat::BGR ||
                 dstFormat == ImageFormat::RGB) {
        out_size = 3 * dsth * dstw;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * dsth * dstw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = dsth * dstw;
      }
      uint8_t* basic_dst = new uint8_t[out_size];
      uint8_t* lite_dst = new uint8_t[out_size];
      Timer t_rotate;
      Timer t_basic, t_lite;
      LOG(INFO) << "baisc resize compute";
      for (int i = 0; i < test_iter; i++) {
        t_basic.Start();
        image_basic_resize(
            src, basic_dst, (ImageFormat)dstFormat, srcw, srch, dstw, dsth);
        t_basic.Stop();
      }
      LOG(INFO) << "image baisc Resize avg time : " << t_basic.LapTimes().Avg()
                << ", min time: " << t_basic.LapTimes().Min()
                << ", max time: " << t_basic.LapTimes().Max();

      LOG(INFO) << "lite resize compute";
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = dsth;
      tparam.ow = dstw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        t_rotate.Start();
        image_preprocess.image_resize(src, lite_dst);
        t_rotate.Stop();
      }
      LOG(INFO) << "image Resize avg time : " << t_rotate.LapTimes().Avg()
                << ", min time: " << t_rotate.LapTimes().Min()
                << ", max time: " << t_rotate.LapTimes().Max();

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      if (FLAGS_check_result) {
        LOG(INFO) << "diff, image Resize size: " << out_size;
        int* diff_v = new int[out_size];
        for (int i = 0; i < out_size; i++) {
          uint8_t a = lite_dst[i];
          uint8_t b = basic_dst[i];
          int diff1 = a - b;  // basic resize and saber resize 在float ->
          // int转换时存在误差，误差范围是{-1, 1}
          int diff = 0;
          if (diff1 < -1 || diff1 > 1) diff = diff1 < 0 ? -diff1 : diff1;
          diff_v[i] = diff;
          if (diff > 1 && max_diff < diff) {
            max_diff = diff;
            printf("i: %d, lite: %d, basic: %d \n", i, a, b);
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = size / srcw;
          printf("din: \n");
          print_int8(src, size, width);
          width = out_size / dstw;
          printf("saber result: \n");
          print_int8(lite_dst, out_size, width);
          printf("basic result: \n");
          print_int8(basic_dst, out_size, width);
          printf("diff result: \n");
          print_int(diff_v, out_size, width);
        }
        delete[] diff_v;
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      LOG(INFO) << "image Resize end";
    }
  }
}

void test_flip(const std::vector<int>& cluster_id,
               const std::vector<int>& thread_num,
               int srcw,
               int srch,
               int dstw,
               int dsth,
               ImageFormat srcFormat,
               ImageFormat dstFormat,
               float rotate,
               FlipParam flip,
               LayoutType layout,
               int test_iter = 10) {
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_tensor_host_rand(src, size);

      int out_size = srch * srcw;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = ceil(1.5 * srch) * srcw;
      } else if (dstFormat == ImageFormat::BGR ||
                 dstFormat == ImageFormat::RGB) {
        out_size = 3 * srch * srcw;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * srch * srcw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
      }
      uint8_t* basic_dst = new uint8_t[out_size];
      uint8_t* lite_dst = new uint8_t[out_size];
      LOG(INFO) << "basic flip compute";
      Timer t_basic, t_lite;
      for (int i = 0; i < test_iter; i++) {
        t_basic.Start();
        image_basic_flip(
            src, basic_dst, (ImageFormat)dstFormat, srcw, srch, flip);
        t_basic.Stop();
      }
      LOG(INFO) << "image baisc flip avg time : " << t_basic.LapTimes().Avg()
                << ", min time: " << t_basic.LapTimes().Min()
                << ", max time: " << t_basic.LapTimes().Max();

      LOG(INFO) << "lite flip compute";
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = srch;
      tparam.ow = srcw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        t_lite.Start();
        image_preprocess.image_flip(src, lite_dst);
        t_lite.Stop();
      }
      LOG(INFO) << "image flip avg time : " << t_lite.LapTimes().Avg()
                << ", min time: " << t_lite.LapTimes().Min()
                << ", max time: " << t_lite.LapTimes().Max();

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      if (FLAGS_check_result) {
        LOG(INFO) << "diff, image flip size: " << out_size;
        uint8_t* diff_v = new uint8_t[out_size];
        for (int i = 0; i < out_size; i++) {
          uint8_t a = lite_dst[i];
          uint8_t b = basic_dst[i];
          uint8_t diff1 = a - b;
          uint8_t diff = diff1 > 0 ? diff1 : -diff1;
          diff_v[i] = diff;
          if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = size / srch;
          printf("din: \n");
          print_int8(src, size, width);
          width = out_size / srch;
          printf("saber result: \n");
          print_int8(lite_dst, out_size, width);
          printf("basic result: \n");
          print_int8(basic_dst, out_size, width);
          printf("diff result: \n");
          print_int8(diff_v, out_size, width);
        }
        delete[] diff_v;
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      LOG(INFO) << "image flip end";
    }
  }
}

void test_rotate(const std::vector<int>& cluster_id,
                 const std::vector<int>& thread_num,
                 int srcw,
                 int srch,
                 int dstw,
                 int dsth,
                 ImageFormat srcFormat,
                 ImageFormat dstFormat,
                 float rotate,
                 FlipParam flip,
                 LayoutType layout,
                 int test_iter = 10) {
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_tensor_host_rand(src, size);

      int out_size = srch * srcw;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = ceil(1.5 * srch) * srcw;
      } else if (dstFormat == ImageFormat::BGR ||
                 dstFormat == ImageFormat::RGB) {
        out_size = 3 * srch * srcw;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * srch * srcw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
      }
      uint8_t* basic_dst = new uint8_t[out_size];
      uint8_t* lite_dst = new uint8_t[out_size];
      LOG(INFO) << "basic rotate compute";
      Timer t_basic, t_lite;
      for (int i = 0; i < test_iter; i++) {
        t_basic.Start();
        image_basic_rotate(
            src, basic_dst, (ImageFormat)dstFormat, srcw, srch, rotate);
        t_basic.Stop();
      }
      LOG(INFO) << "image baisc rotate avg time : " << t_basic.LapTimes().Avg()
                << ", min time: " << t_basic.LapTimes().Min()
                << ", max time: " << t_basic.LapTimes().Max();

      LOG(INFO) << "lite rotate compute";
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = srch;
      tparam.ow = srcw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        t_lite.Start();
        image_preprocess.image_rotate(src, lite_dst);
        t_lite.Stop();
      }
      LOG(INFO) << "image rotate avg time : " << t_lite.LapTimes().Avg()
                << ", min time: " << t_lite.LapTimes().Min()
                << ", max time: " << t_lite.LapTimes().Max();

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      if (FLAGS_check_result) {
        LOG(INFO) << "diff, image rotate size: " << out_size;
        uint8_t* diff_v = new uint8_t[out_size];
        for (int i = 0; i < out_size; i++) {
          uint8_t a = lite_dst[i];
          uint8_t b = basic_dst[i];
          uint8_t diff1 = a - b;
          uint8_t diff = diff1 > 0 ? diff1 : -diff1;
          diff_v[i] = diff;
          if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = size / srch;
          printf("din: \n");
          print_int8(src, size, width);
          width = out_size / srch;
          printf("saber result: \n");
          print_int8(lite_dst, out_size, width);
          printf("basic result: \n");
          print_int8(basic_dst, out_size, width);
          printf("diff result: \n");
          print_int8(diff_v, out_size, width);
        }
        delete[] diff_v;
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      LOG(INFO) << "image rotate end";
    }
  }
}

void test_to_tensor(const std::vector<int>& cluster_id,
                    const std::vector<int>& thread_num,
                    int srcw,
                    int srch,
                    int dstw,
                    int dsth,
                    ImageFormat srcFormat,
                    ImageFormat dstFormat,
                    float rotate,
                    FlipParam flip,
                    LayoutType layout,
                    int test_iter = 10) {
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = new uint8_t[size];
      fill_tensor_host_rand(src, size);

      int out_size = srch * srcw;
      int resize = dstw * dsth;
      if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
        out_size = ceil(1.5 * srch) * srcw;
        resize = ceil(1.5 * dsth) * dstw;
      } else if (dstFormat == ImageFormat::BGR ||
                 dstFormat == ImageFormat::RGB) {
        out_size = 3 * srch * srcw;
        resize = 3 * dsth * dstw;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        out_size = 4 * srch * srcw;
        resize = 4 * dsth * dstw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
        resize = dsth * dstw;
      }
      // out
      std::vector<int64_t> shape_out = {1, 3, dsth, dstw};

      Tensor tensor;
      Tensor tensor_basic;
      tensor.Resize(shape_out);
      tensor_basic.Resize(shape_out);
      tensor.set_precision(PRECISION(kFloat));
      tensor_basic.set_precision(PRECISION(kFloat));

      float means[3] = {127.5f, 127.5f, 127.5f};
      float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};

      Timer t_basic, t_lite;
      LOG(INFO) << "basic to tensor compute: ";
      for (int i = 0; i < test_iter; i++) {
        t_basic.Start();
        image_basic_to_tensor(src,
                              tensor_basic,
                              (ImageFormat)dstFormat,
                              layout,
                              dstw,
                              dsth,
                              means,
                              scales);
        t_basic.Stop();
      }
      LOG(INFO) << "image baisc to_tensor avg time : "
                << t_basic.LapTimes().Avg()
                << ", min time: " << t_basic.LapTimes().Min()
                << ", max time: " << t_basic.LapTimes().Max();

      LOG(INFO) << "lite to_tensor compute";
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = dsth;
      tparam.ow = dstw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;

      Tensor_api dst_tensor(&tensor);
      dst_tensor.Resize(shape_out);

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        t_lite.Start();
        image_preprocess.image_to_tensor(src,
                                         &dst_tensor,
                                         (ImageFormat)dstFormat,
                                         dstw,
                                         dsth,
                                         layout,
                                         means,
                                         scales);
        t_lite.Stop();
      }
      LOG(INFO) << "image tensor avg time : " << t_lite.LapTimes().Avg()
                << ", min time: " << t_lite.LapTimes().Min()
                << ", max time: " << t_lite.LapTimes().Max();

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      if (FLAGS_check_result) {
        max_ratio = 0;
        max_diff = 0;
        LOG(INFO) << "diff, iamge to tensor size: " << tensor.numel();
        const float* ptr_a = tensor.data<float>();
        const float* ptr_b = tensor_basic.data<float>();
        int ss = tensor.numel();
        float* diff_v = new float[ss];
        for (int i = 0; i < ss; i++) {
          int a = ptr_a[i];
          int b = ptr_b[i];
          int diff1 = a - b;
          int diff = 0;
          if (diff1 < -1 || diff1 > 1) diff = diff1 < 0 ? -diff1 : diff1;
          diff_v[i] = diff;
          if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = resize / srch;
          printf("din: \n");
          print_int8(src, resize, width);
          printf("saber result: \n");
          print_fp32(ptr_a, resize, width);
          printf("basic result: \n");
          print_fp32(ptr_b, resize, width);
          printf("diff result: \n");
          print_fp32(diff_v, resize, width);
        }
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
        LOG(INFO) << "iamge to tensor end";
      }
    }
  }
}

void print_info(ImageFormat srcFormat,
                ImageFormat dstFormat,
                int srcw,
                int srch,
                int dstw,
                int dsth,
                float rotate_num,
                int flip_num,
                int layout) {
  paddle::lite::DeviceInfo::Init();
  LOG(INFO) << " input tensor size, num= " << 1 << ", channel= " << 1
            << ", height= " << srch << ", width= " << srcw
            << ", srcFormat= " << (ImageFormat)srcFormat;
  // RGBA = 0, BGRA, RGB, BGR, GRAY, NV21 = 11, NV12,
  if (srcFormat == ImageFormat::NV21) {
    LOG(INFO) << "srcFormat: NV21";
  }
  if (srcFormat == ImageFormat::NV12) {
    LOG(INFO) << "srcFormat: NV12";
  }
  if (srcFormat == ImageFormat::GRAY) {
    LOG(INFO) << "srcFormat: GRAY";
  }
  if (srcFormat == ImageFormat::BGRA) {
    LOG(INFO) << "srcFormat: BGRA";
  }
  if (srcFormat == ImageFormat::BGR) {
    LOG(INFO) << "srcFormat: BGR";
  }
  if (srcFormat == ImageFormat::RGBA) {
    LOG(INFO) << "srcFormat: RGBA";
  }
  if (srcFormat == ImageFormat::RGB) {
    LOG(INFO) << "srcFormat: RGB";
  }
  LOG(INFO) << " output tensor size, num=" << 1 << ", channel=" << 1
            << ", height=" << dsth << ", width=" << dstw
            << ", dstFormat= " << (ImageFormat)dstFormat;

  if (dstFormat == ImageFormat::NV21) {
    LOG(INFO) << "dstFormat: NV21";
  }
  if (dstFormat == ImageFormat::NV12) {
    LOG(INFO) << "dstFormat: NV12";
  }
  if (dstFormat == ImageFormat::GRAY) {
    LOG(INFO) << "dstFormat: GRAY";
  }
  if (dstFormat == ImageFormat::BGRA) {
    LOG(INFO) << "dstFormat: BGRA";
  }
  if (dstFormat == ImageFormat::BGR) {
    LOG(INFO) << "dstFormat: BGR";
  }
  if (dstFormat == ImageFormat::RGBA) {
    LOG(INFO) << "dstFormat: RGBA";
  }
  if (dstFormat == ImageFormat::RGB) {
    LOG(INFO) << "dstFormat: RGB";
  }
  LOG(INFO) << "Rotate = " << rotate_num;
  if (flip_num == -1) {
    LOG(INFO) << "Flip XY";
  } else if (flip_num == 0) {
    LOG(INFO) << "Flip X";
  } else if (flip_num == 1) {
    LOG(INFO) << "Flip Y";
  }
  if (layout == 1) {
    LOG(INFO) << "Layout NCHW";
  } else if (layout == 3) {
    LOG(INFO) << "Layout NHWC";
  }
}
#if 0
TEST(TestImageConvertRand, test_func_image_convert_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 4, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 4, 16, 112, 224}) {
        for (auto rotate : {180}) {
          for (auto flip : {0}) {
            for (auto srcFormat : {12}) {
              for (auto dstFormat : {0, 1, 2, 3}) {
                for (auto layout : {1}) {
                  // RGBA = 0, BGRA, RGB, BGR, GRAY, NV21 = 11, NV12
                  if ((srcFormat == ImageFormat::RGB ||
                      srcFormat == ImageFormat::BGR) &&
                      (dstFormat == ImageFormat::RGBA ||
                       dstFormat == ImageFormat::BGRA)) {
                    continue;  // anakin is not suupport
                  }
                  print_info((ImageFormat)srcFormat,
                            (ImageFormat)dstFormat,
                            w,
                            h,
                            w,
                            h,
                            rotate,
                            flip,
                            layout);
                  test_convert({FLAGS_cluster},
                               {1},
                               w,
                               h,
                               w,
                               h,
                               (ImageFormat)srcFormat,
                               (ImageFormat)dstFormat,
                               rotate,
                               (FlipParam)flip,
                               (LayoutType)layout,
                               FLAGS_repeats);
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 0
TEST(TestImageResizeRand, test_func_image_resize_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {8, 16, 112, 224, 1092}) {
      for (auto h : {4, 16, 112, 224}) {
        for (auto ww : {8, 32, 112}) {
          for (auto hh : {8, 112}) {
            for (auto rotate : {180}) {
              for (auto flip : {0}) {
                for (auto srcFormat : {0, 1, 2, 3, 11, 12}) {
                  for (auto layout : {1}) {
                    auto dstFormat = srcFormat;
                    print_info((ImageFormat)srcFormat,
                                (ImageFormat)dstFormat,
                                w,
                                h,
                                ww,
                                hh,
                                rotate,
                                flip,
                                layout);
                    test_resize({FLAGS_cluster},
                                {1},
                                w,
                                h,
                                ww,
                                hh,
                                (ImageFormat)srcFormat,
                                (ImageFormat)dstFormat,
                                rotate,
                                (FlipParam)flip,
                                (LayoutType)layout,
                                FLAGS_repeats);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 1
TEST(TestImageFlipRand, test_func_image_flip_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 8, 16, 112, 224, 1080}) {
      for (auto h : {1, 16, 112, 224}) {
        for (auto rotate : {90}) {
          for (auto flip : {-1, 0, 1}) {
            for (auto srcFormat : {0, 1, 2, 3}) {
              for (auto layout : {1}) {
                auto dstFormat = srcFormat;
                print_info((ImageFormat)srcFormat,
                           (ImageFormat)dstFormat,
                           w,
                           h,
                           w,
                           h,
                           rotate,
                           flip,
                           layout);
                test_flip({FLAGS_cluster},
                          {1},
                          w,
                          h,
                          w,
                          h,
                          (ImageFormat)srcFormat,
                          (ImageFormat)dstFormat,
                          rotate,
                          (FlipParam)flip,
                          (LayoutType)layout,
                          FLAGS_repeats);
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 1
TEST(TestImageRotateRand, test_func_image_rotate_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 16, 112, 224}) {
        for (auto rotate : {90, 180, 270}) {
          for (auto flip : {0}) {
            for (auto srcFormat : {0, 1, 2, 3}) {
              for (auto layout : {1}) {
                auto dstFormat = srcFormat;
                print_info((ImageFormat)srcFormat,
                           (ImageFormat)dstFormat,
                           w,
                           h,
                           w,
                           h,
                           rotate,
                           flip,
                           layout);
                test_rotate({FLAGS_cluster},
                            {1},
                            w,
                            h,
                            w,
                            h,
                            (ImageFormat)srcFormat,
                            (ImageFormat)dstFormat,
                            rotate,
                            (FlipParam)flip,
                            (LayoutType)layout,
                            FLAGS_repeats);
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 1
TEST(TestImageToTensorRand, test_func_image_to_tensor_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 16, 112, 224}) {
        for (auto rotate : {90}) {
          for (auto flip : {0}) {
            for (auto srcFormat : {0, 1, 2, 3}) {
              for (auto layout : {1}) {
                auto dstFormat = srcFormat;
                print_info((ImageFormat)srcFormat,
                           (ImageFormat)dstFormat,
                           w,
                           h,
                           w,
                           h,
                           rotate,
                           flip,
                           layout);
                test_to_tensor({FLAGS_cluster},
                               {1},
                               w,
                               h,
                               w,
                               h,
                               (ImageFormat)srcFormat,
                               (ImageFormat)dstFormat,
                               rotate,
                               (FlipParam)flip,
                               (LayoutType)layout,
                               FLAGS_repeats);
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 1
TEST(TestImageConvertCustom, test_func_image_preprocess_custom) {
  LOG(INFO) << "print info";
  print_info((ImageFormat)FLAGS_srcFormat,
             (ImageFormat)FLAGS_dstFormat,
             FLAGS_srcw,
             FLAGS_srch,
             FLAGS_dstw,
             FLAGS_dsth,
             FLAGS_angle,
             FLAGS_flip_num,
             FLAGS_layout);
  test_convert({FLAGS_cluster},
               {1},
               FLAGS_srcw,
               FLAGS_srch,
               FLAGS_dstw,
               FLAGS_dsth,
               (ImageFormat)FLAGS_srcFormat,
               (ImageFormat)FLAGS_dstFormat,
               FLAGS_angle,
               (FlipParam)FLAGS_flip_num,
               (LayoutType)FLAGS_layout,
               FLAGS_repeats);

  test_resize({FLAGS_cluster},
              {1},
              FLAGS_srcw,
              FLAGS_srch,
              FLAGS_dstw,
              FLAGS_dsth,
              (ImageFormat)FLAGS_dstFormat,
              (ImageFormat)FLAGS_dstFormat,
              FLAGS_angle,
              (FlipParam)FLAGS_flip_num,
              (LayoutType)FLAGS_layout,
              FLAGS_repeats);
  test_flip({FLAGS_cluster},
            {1},
            FLAGS_srcw,
            FLAGS_srch,
            FLAGS_dstw,
            FLAGS_dsth,
            (ImageFormat)FLAGS_dstFormat,
            (ImageFormat)FLAGS_dstFormat,
            FLAGS_angle,
            (FlipParam)FLAGS_flip_num,
            (LayoutType)FLAGS_layout,
            FLAGS_repeats);
  test_rotate({FLAGS_cluster},
              {1},
              FLAGS_srcw,
              FLAGS_srch,
              FLAGS_dstw,
              FLAGS_dsth,
              (ImageFormat)FLAGS_dstFormat,
              (ImageFormat)FLAGS_dstFormat,
              FLAGS_angle,
              (FlipParam)FLAGS_flip_num,
              (LayoutType)FLAGS_layout,
              FLAGS_repeats);
  test_to_tensor({FLAGS_cluster},
                 {1},
                 FLAGS_srcw,
                 FLAGS_srch,
                 FLAGS_dstw,
                 FLAGS_dsth,
                 (ImageFormat)FLAGS_dstFormat,
                 (ImageFormat)FLAGS_dstFormat,
                 FLAGS_angle,
                 (FlipParam)FLAGS_flip_num,
                 (LayoutType)FLAGS_layout,
                 FLAGS_repeats);
}
#endif
#endif
