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
#include "lite/tests/cv/cv_basic.h"
#include "lite/tests/utils/timer.h"
// #include "lite/utils/cv/image_convert.h"
#ifdef LITE_WITH_ARM
#include "lite/utils/cv/image_convert.h"
#include "lite/utils/cv/image_preprocess.h"
#endif  // LITE_WITH_ARM

// DEFINE_int32(cluster, 3, "cluster id");
// DEFINE_int32(threads, 1, "threads num");
// DEFINE_int32(warmup, 0, "warmup times");
// DEFINE_int32(repeats, 1, "repeats times");
// DEFINE_bool(basic_test, false, "do all tests");
// DEFINE_bool(check_result, true, "check the result");

// DEFINE_int32(srcFormat, 0, "input image format");
// DEFINE_int32(dstFormat, 1, "output image format");
// DEFINE_int32(srch, 1920, "input height");
// DEFINE_int32(srcw, 1080, "input width");
// DEFINE_int32(dsth, 960, "output height");
// DEFINE_int32(dstw, 540, "output width");
// DEFINE_float(angle, 90, "rotate angel");
// DEFINE_int32(flip_num, 0, "flip x");
// DEFINE_int32(layout, 0, "layout chw");
int FLAGS_srch = 1920;
int FLAGS_srcw = 1080;
int FLAGS_dstw = 960;
int FLAGS_dsth = 540;
int FLAGS_angle = 90;
int FLAGS_flip_num = 1;
int FLAGS_check_result = 1;
int FLAGS_cluster = 3;
int FLAGS_threads = 1;
int FLAGS_warmup = 0;
int FLAGS_repeats = 1;
int FLAGS_srcFormat = 0;
int FLAGS_dstFormat = 1;
int FLAGS_layout = 0;
bool FLAGS_basic_test = 1;

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::utils::cv::LayOut LayOut;
typedef paddle::lite::utils::cv::ImageConvert ImageConvert;
typedef paddle::lite::utils::cv::Transform Transform;
typedef paddle::lite::utils::cv::TransParam TransParam;
typedef paddle::lite::utils::cv::ImageTransform ImageTransform;
typedef paddle::lite::utils::cv::ImagePreprocess ImagePreprocess;
typedef paddle::lite::Tensor Tensor;
using paddle::lite::Timer;

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

void print_ff(const float* ptr, int size, int width) {
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
void test_img(const std::vector<int>& cluster_id,
              const std::vector<int>& thread_num,
              int srcw,
              int srch,
              int dstw,
              int dsth,
              ImageFormat srcFormat,
              ImageFormat dstFormat,
              float rotate,
              FlipParam flip,
              LayOut layout,
              int test_iter = 1) {
  LOG(INFO) << "test_func_image_preprocess start";
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);

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

      LOG(INFO) << "Rotate = " << rotate << ", Flip = " << flip
                << ", Layout = " << layout;

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
      // uint8_t* convert_buff = new uint8_t[out_size * 2];
      uint8_t* basic_dst = new uint8_t[out_size];
      uint8_t* lite_dst = new uint8_t[out_size];

      // resize
      uint8_t* resize_basic = new uint8_t[resize];
      uint8_t* resize_tmp = new uint8_t[resize];

      uint8_t* tv_out_ratote_basic = new uint8_t[resize];
      uint8_t* tv_out_ratote = new uint8_t[resize];

      uint8_t* tv_out_flip_basic = new uint8_t[resize];
      uint8_t* tv_out_flip = new uint8_t[resize];

      // Shape shape_out(1, 3, dsth, dstw);
      std::vector<int64_t> shape_out = {1, 3, dsth, dstw};

      Tensor tensor;
      Tensor tensor_basic;
      tensor.Resize(shape_out);
      tensor_basic.Resize(shape_out);
      tensor.set_precision(PRECISION(kFloat));
      tensor_basic.set_precision(PRECISION(kFloat));
      // fill_tensor_const(tensor, 0.f);
      // fill_tensor_const(tensor_basic, 0.f);
      float means[3] = {127.5f, 127.5f, 127.5f};
      float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};

      if (FLAGS_check_result) {
        LOG(INFO) << "image convert basic compute";
        image_convert_basic(src,
                            basic_dst,
                            (ImageFormat)srcFormat,
                            (ImageFormat)dstFormat,
                            srcw,
                            srch,
                            out_size);

        LOG(INFO) << "image resize basic compute";
        image_resize_basic(basic_dst,
                           resize_basic,
                           (ImageFormat)dstFormat,
                           srcw,
                           srch,
                           dstw,
                           dsth);

        LOG(INFO) << "image rotate basic compute";
        image_rotate_basic(resize_basic,
                           tv_out_ratote_basic,
                           (ImageFormat)dstFormat,
                           dstw,
                           dsth,
                           rotate);

        LOG(INFO) << "image flip basic compute";
        image_flip_basic(resize_basic,
                         tv_out_flip_basic,
                         (ImageFormat)dstFormat,
                         dstw,
                         dsth,
                         flip);

        LOG(INFO) << "image to tensor basic compute";
        image_to_tensor_basic(resize_basic,
                              &tensor_basic,
                              (ImageFormat)dstFormat,
                              layout,
                              dstw,
                              dsth,
                              means,
                              scales);
      }

      Timer t1;

      LOG(INFO) << "saber cv compute";
      double to = 0;
      double min_time = 100000;
      TransParam tparam;
      tparam.ih = srch;
      tparam.iw = srcw;
      tparam.oh = dsth;
      tparam.ow = dstw;
      tparam.flip_param = flip;
      tparam.rotate_param = rotate;
      std::vector<Transform> v_trans;
      v_trans.push_back((Transform)0);
      v_trans.push_back((Transform)1);
      tparam.v_trans = v_trans;

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);
      ImageTransform image_trans;

      for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();

        LOG(INFO) << "image convert saber compute";
        image_preprocess.imageCovert(
            src, lite_dst, (ImageFormat)srcFormat, (ImageFormat)dstFormat);

        LOG(INFO) << "image resize saber compute";
        image_preprocess.imageResize(
            lite_dst, resize_tmp, (ImageFormat)dstFormat);

        LOG(INFO) << "image rotate saber compute";
        // image_preprocess.imageTransform(resize_tmp, tv_out_flip_basic);
        image_trans.rotate(resize_tmp,
                           tv_out_ratote,
                           (ImageFormat)dstFormat,
                           dstw,
                           dsth,
                           rotate);

        LOG(INFO) << "image flip saber compute";
        // imFLAGS_preprocess.imageTransform(resize_tmp, tv_out_flip);
        image_trans.flip(
            resize_tmp, tv_out_flip, (ImageFormat)dstFormat, dstw, dsth, flip);

        LOG(INFO) << "image to tensor compute";
        image_preprocess.image2Tensor(
            resize_tmp, &tensor, layout, means, scales);

        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
          min_time = tdiff;
        }
      }
      LOG(INFO) << "image trans total time : " << to << ",  avg time : %"
                << to / test_iter;
      // print_tensor(tout);
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
      if (FLAGS_check_result) {
        max_ratio = 0;
        max_diff = 0;
        // const double eps = 1e-6f;
        int* diff_v = new int[resize];
        LOG(INFO) << "diff, image resize size: " << resize;
        for (int i = 0; i < resize; i++) {
          uint8_t a = resize_tmp[i];
          uint8_t b = resize_basic[i];
          int diff1 = a - b;
          int diff = 0;  // basic resize and saber resize 在float ->
                         // int转换时存在误差，误差范围是{-1, 1}
          if (diff1 < -1 || diff1 > 1) diff = diff1 < 0 ? -diff1 : diff1;
          diff_v[i] = diff;
          if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
          }
        }
        if (std::abs(max_ratio) >= 1e-5f) {
          int width = out_size / srch;
          printf("din: \n");
          print_int8(lite_dst, out_size, width);
          width = resize / dsth;
          printf("saber result: \n");
          print_int8(resize_tmp, resize, width);
          printf("basic result: \n");
          print_int8(resize_basic, resize, width);
          printf("diff result: \n");
          print_int(diff_v, resize, width);
        }
        delete[] diff_v;
        // printf("\n");
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      delete[] lite_dst;
      delete[] basic_dst;
      LOG(INFO) << "image resize end";

      if (FLAGS_check_result) {
        max_ratio = 0;
        max_diff = 0;
        // const double eps = 1e-6f;
        int* diff_v = new int[resize];
        LOG(INFO) << "diff, image rotate size: " << resize;
        for (int i = 0; i < resize; i++) {
          int a = tv_out_ratote[i];
          int b = tv_out_ratote_basic[i];
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
          int width = resize / dsth;
          printf("din: \n");
          print_int8(resize_tmp, resize, width);
          printf("saber result: \n");
          print_int8(tv_out_ratote, resize, width);
          printf("basic result: \n");
          print_int8(tv_out_ratote_basic, resize, width);
          printf("diff result: \n");
          print_int(diff_v, resize, width);
        }
        delete[] diff_v;
        // printf("\n");
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      delete[] tv_out_ratote;
      delete[] tv_out_ratote_basic;
      LOG(INFO) << "image rotate end";

      if (FLAGS_check_result) {
        max_ratio = 0;
        max_diff = 0;
        // const double eps = 1e-6f;
        int* diff_v = new int[resize];
        LOG(INFO) << "diff, image flip size: " << resize;
        for (int i = 0; i < resize; i++) {
          int a = tv_out_flip[i];
          int b = tv_out_flip_basic[i];
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
          int width = resize / dsth;
          printf("din: \n");
          print_int8(resize_tmp, resize, width);
          printf("saber result: \n");
          print_int8(tv_out_flip, resize, width);
          printf("basic result: \n");
          print_int8(tv_out_flip_basic, resize, width);
          printf("diff result: \n");
          print_int(diff_v, resize, width);
        }
        delete[] diff_v;
        // printf("\n");
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
      }
      delete[] tv_out_flip;
      delete[] tv_out_flip_basic;
      delete[] resize_tmp;
      delete[] resize_basic;
      LOG(INFO) << "image flip  end";

      if (FLAGS_check_result) {
        max_ratio = 0;
        max_diff = 0;
        // const double eps = 1e-6f;
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
          print_int8(resize_tmp, resize, width);
          printf("saber result: \n");
          print_ff(ptr_a, resize, width);
          printf("basic result: \n");
          print_ff(ptr_b, resize, width);
          printf("diff result: \n");
          print_ff(diff_v, resize, width);
        }
        // printf("\n");
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        bool rst = std::abs(max_ratio) < 1e-5f;
        CHECK_EQ(rst, true) << "compute result error";
        LOG(INFO) << "iamge to tensor end";
      }
    }
  }
}

#if 1
TEST(TestImageConvertRand, test_func_image_convert_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 4, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 4, 16, 112, 224}) {
        for (auto ww : {66}) {
          for (auto hh : {12}) {
            for (auto rotate : {180}) {
              for (auto flip : {0}) {
                for (auto srcFormat : {0, 1, 2, 3, 4, 11, 12}) {
                  for (auto dstFormat : {0, 1, 2, 3}) {
                    for (auto layout : {0}) {
                      if ((dstFormat == ImageFormat::GRAY &&
                           (srcFormat == ImageFormat::RGBA ||
                            srcFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::GRAY &&
                           (dstFormat == ImageFormat::RGBA ||
                            dstFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::NV12 ||
                           srcFormat == ImageFormat::NV21) &&
                              (dstFormat == ImageFormat::GRAY ||
                               dstFormat == ImageFormat::RGBA ||
                               dstFormat == ImageFormat::BGRA)) {
                        continue;
                      }
                      if (srcFormat == ImageFormat::NV12 ||
                          srcFormat == ImageFormat::NV21) {
                        if (w % 2) {  // is not ou shu, two line y == one line
                                      // uv
                          continue;
                        }
                      }
                      test_img({FLAGS_cluster},
                               {1, 2, 4},
                               w,
                               h,
                               ww,
                               hh,
                               (ImageFormat)srcFormat,
                               (ImageFormat)dstFormat,
                               rotate,
                               (FlipParam)flip,
                               (LayOut)layout);
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
}
#endif
#if 1
TEST(TestImageConvertRand, test_func_image_resize_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 4, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 4, 16, 112, 224}) {
        for (auto ww : {1, 2, 8, 32, 112}) {
          for (auto hh : {1, 2, 8, 112}) {
            for (auto rotate : {180}) {
              for (auto flip : {0}) {
                for (auto srcFormat : {0, 1, 2, 3, 4, 11, 12}) {
                  for (auto dstFormat : {0, 1, 2, 3}) {
                    for (auto layout : {0}) {
                      if (dstFormat == ImageFormat::NV12 ||
                          dstFormat == ImageFormat::NV21 ||
                          (dstFormat == ImageFormat::GRAY &&
                           (srcFormat == ImageFormat::RGBA ||
                            srcFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::GRAY &&
                           (dstFormat == ImageFormat::RGBA ||
                            dstFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::NV12 ||
                           srcFormat == ImageFormat::NV21) &&
                              (dstFormat == ImageFormat::GRAY ||
                               dstFormat == ImageFormat::RGBA ||
                               dstFormat == ImageFormat::BGRA)) {
                        continue;
                      }
                      if (srcFormat == ImageFormat::NV12 ||
                          srcFormat == ImageFormat::NV21) {
                        if (w % 2) {  // is not ou shu, two line y == one line
                                      // uv
                          continue;
                        }
                      }
                      test_img({FLAGS_cluster},
                               {1, 2, 4},
                               w,
                               h,
                               ww,
                               hh,
                               (ImageFormat)srcFormat,
                               (ImageFormat)dstFormat,
                               rotate,
                               (FlipParam)flip,
                               (LayOut)layout);
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
}
#endif
#if 1
TEST(TestImageConvertRand, test_func_image_trans_preprocess) {
  if (FLAGS_basic_test) {
    for (auto w : {1, 8, 16, 112, 224, 1092}) {
      for (auto h : {1, 16, 112, 224}) {
        for (auto ww : {32, 112}) {
          for (auto hh : {32, 112}) {
            for (auto rotate : {90, 180, 270}) {
              for (auto flip : {0, 1, 2}) {
                for (auto srcFormat : {11}) {
                  for (auto dstFormat : {3}) {
                    for (auto layout : {0, 1}) {
                      if (dstFormat == ImageFormat::NV12 ||
                          dstFormat == ImageFormat::NV21 ||
                          (dstFormat == ImageFormat::GRAY &&
                           (srcFormat == ImageFormat::RGBA ||
                            srcFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::GRAY &&
                           (dstFormat == ImageFormat::RGBA ||
                            dstFormat == ImageFormat::BGRA)) ||
                          (srcFormat == ImageFormat::NV12 ||
                           srcFormat == ImageFormat::NV21) &&
                              (dstFormat == ImageFormat::GRAY ||
                               dstFormat == ImageFormat::RGBA ||
                               dstFormat == ImageFormat::BGRA)) {
                        continue;
                      }
                      if (srcFormat == ImageFormat::NV12 ||
                          srcFormat == ImageFormat::NV21) {
                        if (w % 2) {  // is not ou shu, two line y == one line
                                      // uv
                          continue;
                        }
                      }
                      test_img({FLAGS_cluster},
                               {1, 2, 4},
                               w,
                               h,
                               ww,
                               hh,
                               (ImageFormat)srcFormat,
                               (ImageFormat)dstFormat,
                               rotate,
                               (FlipParam)flip,
                               (LayOut)layout);
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
}
#endif
#if 1
TEST(TestImageConvertCustom, test_func_image_preprocess_custom) {
  test_img({FLAGS_cluster},
           {1, 2, 4},
           FLAGS_srcw,
           FLAGS_srch,
           FLAGS_dstw,
           FLAGS_dsth,
           (ImageFormat)FLAGS_srcFormat,
           (ImageFormat)FLAGS_dstFormat,
           FLAGS_angle,
           (FlipParam)FLAGS_flip_num,
           (LayOut)FLAGS_layout);
}
#endif
#endif
