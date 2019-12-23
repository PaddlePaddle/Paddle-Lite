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

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
// #include "paddle_image_preprocess.h"
// #include "time.h"

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::utils::cv::TransParam TransParam;
typedef paddle::lite::utils::cv::ImagePreprocess ImagePreprocess;
typedef paddle::lite_api::Tensor Tensor_api;
typedef paddle::lite_api::DataLayoutType LayoutType;

void test_img(std::vectors<int> cluster_id,
              std::vectors<int> thread_num,
              std::string img_path,
              std::string dst_path,
              ImageFormat srcFormat,
              ImageFormat dstFormat,
              int width,
              int height,
              float rotate,
              int flip,
              int test_iter = 1) {
  // init
  paddle::lite::DeviceInfo::Init();
  int srch = im.rows;
  int srcw = img.cols;
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      LOG(INFO) << "cluster: " << cls << ", threads: " << th;

      /*
        imread(img_path, param)
        IMREAD_UNCHANGED(<0) 表示加载原图，不做任何改变
        IMREAD_GRAYSCALE ( 0)表示把原图作为灰度图像加载进来
        IMREAD_COLOR (>0) 表示把原图作为RGB图像加载进来
      */
      cv::Mat img;
      if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
        img = imread(img_path, cv::IMREAD_COLOR);
      } else if (srcFormat == ImageFormat::GRAY) {
        img = imread(img_path, cv::IMREAD_GRAYSCALE);
      } else {
        printf("this format %d does not support \n", srcFormat);
        return;
      }
      if (img.empty()) {
        LOG(FATAL) << "opencv read image " << img_name.c_str() << " failed";
      }
      int srch = img.rows;
      int srcw = img.cols;
      int dsth = height;
      int dstw = width;

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
                << ", Layout = " << static_cast<int>(layout);

      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
        size = ceil(1.5 * srch) * srcw;
      } else if (srcFormat == ImageFormat::BGRA ||
                 srcFormat == ImageFormat::RGBA) {
        size = 4 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = img.data;

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
      uint8_t* lite_dst = new uint8_t[out_size];
      uint8_t* resize_tmp = new uint8_t[resize];
      uint8_t* tv_out_ratote = new uint8_t[size];
      uint8_t* tv_out_flip = new uint8_t[size];
      std::vector<int64_t> shape_out = {1, 3, dsth, dstw};

      Tensor tensor;
      Tensor tensor_basic;
      tensor.Resize(shape_out);
      tensor_basic.Resize(shape_out);
      tensor.set_precision(PRECISION(kFloat));
      tensor_basic.set_precision(PRECISION(kFloat));

      float means[3] = {127.5f, 127.5f, 127.5f};
      float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
      LOG(INFO) << "opencv compute";
      cv::Mat im_convert;
      cv::Mat im_resize;
      cv::Mat im_rotate;
      cv::Mat im_flip;
      double to_1 = 0;
      double to_2 = 0;
      double to_3 = 0;
      double to_4 = 0;
      double to1 = 0;
      for (int i = 0; i < test_iter; i++) {
        clock_t start = clock();
        clock_t begin = clock();
        // convert bgr-gray
        if (dstFormat == ImageFormat::GRAY) {
          cv::cvtColor(img, im_convert, cv::COLOR_BGR2GRAY)
        } else {
          cv::cvtColor(img, im_convert, cv::COLOR_GRAY2BGR);
        }
        clock_t end = clock();
        to_1 += (end - begin);

        begin = clock();
        // resize default linear
        cv::resize(img, im_resize, cv::Size(dstw, dsth), 0.f, 0.f);
        end = clock();
        to_2 += (end - begin);

        begin = clock();
        // rotate 90
        cv::transpose(img, im_rotate);
        end = clock();
        to_3 += (end - begin);

        begin = clock();
        // flip
        cv::flip(img, im_flip, flip);
        end = clock();
        to_4 += (end - begin);
        clock_t ovet = clock();
        to1 += (ovet - start);
      }

      LOG(INFO) << "Paddle-lite compute";
      double lite_to = 0;
      double lite_to_1 = 0;
      double lite_to_2 = 0;
      double lite_to_3 = 0;
      double lite_to_4 = 0;
      double lite_to_5 = 0;
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
        clock_t start = clock();
        // LOG(INFO) << "image convert saber compute";
        clock_t begin = clock();
        image_preprocess.imageCovert(src, lite_dst);
        clock_t end = clock();
        lite_to_1 += (end - start);

        // LOG(INFO) << "image resize saber compute";
        begin = clock();
        image_preprocess.imageResize(
            src, resize_tmp, (ImageFormat)srcFormat, srcw, srch, dstw, dsth);
        end = clock();
        lite_to_2 += (end - start);

        // LOG(INFO) << "image rotate saber compute";
        begin = clock();
        image_preprocess.imageRotate(
            src, tv_out_ratote, (ImageFormat)srcFormat, srcw, srch, 90);
        end = clock();
        lite_to_3 += (end - start);

        // LOG(INFO) << "image flip saber compute";
        begin = clock();
        image_preprocess.imageFlip(
            resize_tmp, tv_out_flip, (ImageFormat)srcFormat, srcw, srch, flip);
        end = clock();
        lite_to_4 += (end - start);
        clcok_t over = clock();
        lite_to += (over - start);

        // LOG(INFO) << "image to tensor compute";
        begin = clock();
        image_preprocess.image2Tensor(resize_tmp,
                                      &dst_tensor,
                                      (ImageFormat)dstFormat,
                                      dstw,
                                      dsth,
                                      layout,
                                      means,
                                      scales);
        end = clock();
        lite_to_5 += (end - start);
      }
      to_1 = 1000 * to_1 / CLOCKS_PER_SEC;
      to_2 = 1000 * to_2 / CLOCKS_PER_SEC;
      to_3 = 1000 * to_3 / CLOCKS_PER_SEC;
      to_4 = 1000 * to_4 / CLOCKS_PER_SEC;
      to1 = 1000 * to1 / CLOCKS_PER_SEC;
      LOG(INFO) << "opencv convert run time: " << to_1
                << "ms, avg: " << to_1 / test_iter;
      LOG(INFO) << "opencv resize run time: " << to_2
                << "ms, avg: " << to_2 / test_iter;
      LOG(INFO) << "opencv rotate run time: " << to_3
                << "ms, avg: " << to_3 / test_iter;
      LOG(INFO) << "opencv flip  time: " << to_4
                << "ms, avg: " << to_4 / test_iter;
      LOG(INFO) << "opencv total run time: " << to1
                << "ms, avg: " << to1 / test_iter;
      LOG(INFO) << "------";

      lite_to_1 = 1000 * lite_to_1 / CLOCKS_PER_SEC;
      lite_to_2 = 1000 * lite_to_2 / CLOCKS_PER_SEC;
      lite_to_3 = 1000 * lite_to_3 / CLOCKS_PER_SEC;
      lite_to_4 = 1000 * lite_to_4 / CLOCKS_PER_SEC;
      lite_to_5 = 1000 * lite_to_5 / CLOCKS_PER_SEC;
      lite_to = 1000 * lite_to / CLOCKS_PER_SEC;
      LOG(INFO) << "lite convert run time: " << lite_to_1
                << "ms, avg: " << lite_to_1 / test_iter;
      LOG(INFO) << "lite resize run time: " << lite_to_2
                << "ms, avg: " << lite_to_2 / test_iter;
      LOG(INFO) << "lite rotate run time: " << lite_to_3
                << "ms, avg: " << lite_to_3 / test_iter;
      LOG(INFO) << "lite flip  time: " << lite_to_4
                << "ms, avg: " << lite_to_4 / test_iter;
      LOG(INFO) << "lite img2tensor  time: " << lite_to_5
                << "ms, avg: " << lite_to_5 / test_iter;
      LOG(INFO) << "lite total run time: " << lite_to
                << "ms, avg: " << lite_to / test_iter;
      LOG(INFO) << "------";

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      // save_img
      LOG(INFO) << "write image: ";
      std::string resize_name = dst_path + "/resize.jpeg";
      std::string convert_name = dst_path + "/convert.jpeg";
      std::string rotate_name = dst_path + "/rotate.jpeg";
      std::string flip_name = dst_path + "/flip.jpeg";
      cv::imwrite(convert_name, lite_dst);
      cv::imwrite(resize_name, resize_tmp);
      cv::imwrite(rotate_name, tv_out_ratote);
      cv::imwrite(flip_name, tv_out_flip);
      delete[] lite_dst;
      delete[] resize_tmp;
      delete[] tv_out_ratote;
      delete[] tv_out_flip;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " image_path dst_apth srcFormat dstFormat width height\n";
    exit(1);
  }
  std::string image_path = argv[1];
  std::string dst_path = argv[2];
  int srcFormat = atoi(argv[3]);
  int dstFormat = atoi(argv[4]);
  int width = atoi(argv[5]);
  int height = atoi(argv[6]);
  int flip = -1;
  float rotate = 90;
  if (argc > 7) {
    flip = atoi(argv[7]);
  }
  if (argc > 8) {
    rotate = atoi(argv[8]);
  }

  test_img({3},
           {1, 2, 4},
           image_path,
           dst_path,
           srcFormat,
           dstFormat,
           width,
           height,
           rotate,
           flip,
           20);
  return 0;
}
