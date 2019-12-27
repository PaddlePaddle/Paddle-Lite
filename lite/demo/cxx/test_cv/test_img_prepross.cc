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
#include "paddle_api.h"               // NOLINT
#include "paddle_image_preprocess.h"  // NOLINT
#include "time.h"                     // NOLINT
typedef paddle::lite_api::Tensor Tensor;
typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::utils::cv::TransParam TransParam;
typedef paddle::lite::utils::cv::ImagePreprocess ImagePreprocess;
typedef paddle::lite_api::DataLayoutType LayoutType;
using namespace paddle::lite_api;  // NOLINT

void fill_with_mat(cv::Mat& mat, uint8_t* src) {  // NOLINT
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      int tmp = (i * mat.cols + j) * 3;
      cv::Vec3b& rgb = mat.at<cv::Vec3b>(i, j);
      rgb[0] = src[tmp];
      rgb[1] = src[tmp + 1];
      rgb[2] = src[tmp + 2];
    }
  }
}
void test_img(std::vector<int> cluster_id,
              std::vector<int> thread_num,
              std::string img_path,
              std::string dst_path,
              ImageFormat srcFormat,
              ImageFormat dstFormat,
              int width,
              int height,
              float rotate,
              FlipParam flip,
              LayoutType layout,
              std::string model_dir,
              int test_iter = 1) {
  // init
  // paddle::lite::DeviceInfo::Init();
  // read img and pre-process
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  float means[3] = {0.485f, 0.456f, 0.406f};
  float scales[3] = {0.229f, 0.224f, 0.225f};
  int srch = img.rows;
  int srcw = img.cols;
  for (auto& cls : cluster_id) {
    for (auto& th : thread_num) {
      std::cout << "cluster: " << cls << ", threads: " << th << std::endl;
      // 1. Set MobileConfig
      MobileConfig config;
      config.set_model_dir(model_dir);
      config.set_power_mode((PowerMode)cls);
      config.set_threads(th);
      std::cout << "model: " << model_dir;

      // 2. Create PaddlePredictor by MobileConfig
      std::shared_ptr<PaddlePredictor> predictor =
          CreatePaddlePredictor<MobileConfig>(config);

      // 3. Prepare input data from image
      std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));

      // read img and pre-process
      float means[3] = {0.485f, 0.456f, 0.406f};
      float scales[3] = {0.229f, 0.224f, 0.225f};
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
        std::cout << "opencv read image " << img_path.c_str() << " failed"
                  << std::endl;
        return;
      }
      int srch = img.rows;
      int srcw = img.cols;
      int dsth = height;
      int dstw = width;

      std::cout << " input tensor size, num= " << 1 << ", channel= " << 1
                << ", height= " << srch << ", width= " << srcw
                << ", srcFormat= " << (ImageFormat)srcFormat << std::endl;
      // RGBA = 0, BGRA, RGB, BGR, GRAY, NV21 = 11, NV12,
      if (srcFormat == ImageFormat::GRAY) {
        std::cout << "srcFormat: GRAY" << std::endl;
      }
      if (srcFormat == ImageFormat::BGR) {
        std::cout << "srcFormat: BGR" << std::endl;
      }
      if (srcFormat == ImageFormat::RGB) {
        std::cout << "srcFormat: RGB" << std::endl;
      }
      std::cout << " output tensor size, num=" << 1 << ", channel=" << 1
                << ", height=" << dsth << ", width=" << dstw
                << ", dstFormat= " << (ImageFormat)dstFormat << std::endl;

      if (dstFormat == ImageFormat::GRAY) {
        std::cout << "dstFormat: GRAY" << std::endl;
      }
      if (dstFormat == ImageFormat::BGR) {
        std::cout << "dstFormat: BGR" << std::endl;
      }
      if (dstFormat == ImageFormat::RGB) {
        std::cout << "dstFormat: RGB" << std::endl;
      }

      std::cout << "Rotate = " << rotate << ", Flip = " << flip
                << ", Layout = " << static_cast<int>(layout) << std::endl;
      if (static_cast<int>(layout) != 1 && static_cast<int>(layout) != 3) {
        std::cout << "this layout" << static_cast<int>(layout)
                  << " is no support" << std::endl;
      }
      int size = 3 * srch * srcw;
      if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
        size = 3 * srch * srcw;
      } else if (srcFormat == ImageFormat::GRAY) {
        size = srch * srcw;
      }
      uint8_t* src = img.data;

      int out_size = srch * srcw;
      int resize = dstw * dsth;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        out_size = 3 * srch * srcw;
        resize = 3 * dsth * dstw;
      } else if (dstFormat == ImageFormat::GRAY) {
        out_size = srch * srcw;
        resize = dsth * dstw;
      }
      // out
      uint8_t* lite_dst = new uint8_t[out_size];
      uint8_t* resize_tmp = new uint8_t[resize];
      uint8_t* tv_out_ratote = new uint8_t[out_size];
      uint8_t* tv_out_flip = new uint8_t[out_size];
      std::vector<int64_t> shape_out = {1, 3, srch, srcw};

      input_tensor->Resize(shape_out);
      Tensor dst_tensor = *input_tensor;
      std::cout << "opencv compute" << std::endl;
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
        if (dstFormat == srcFormat) {
          im_convert = img;
        } else if (dstFormat == ImageFormat::BGR &&
                   srcFormat == ImageFormat::GRAY) {
          cv::cvtColor(img, im_convert, cv::COLOR_GRAY2BGR);
        } else if (srcFormat == ImageFormat::BGR &&
                   dstFormat == ImageFormat::GRAY) {
          cv::cvtColor(img, im_convert, cv::COLOR_BGR2GRAY);
        } else if (dstFormat == srcFormat) {
          printf("convert format error \n");
          return;
        }
        clock_t end = clock();
        to_1 += (end - begin);

        begin = clock();
        // resize default linear
        cv::resize(im_convert, im_resize, cv::Size(dstw, dsth), 0.f, 0.f);
        end = clock();
        to_2 += (end - begin);

        begin = clock();
        // rotate 90
        cv::transpose(im_convert, im_rotate);
        end = clock();
        to_3 += (end - begin);

        begin = clock();
        // flip
        cv::flip(im_convert, im_flip, flip);
        end = clock();
        to_4 += (end - begin);
        clock_t ovet = clock();
        to1 += (ovet - start);
      }

      std::cout << "Paddle-lite compute" << std::endl;
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

      ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);

      for (int i = 0; i < test_iter; ++i) {
        clock_t start = clock();
        clock_t begin = clock();
        image_preprocess.imageConvert(src, lite_dst);
        clock_t end = clock();
        lite_to_1 += (end - begin);

        begin = clock();
        image_preprocess.imageResize(lite_dst, resize_tmp);
        end = clock();
        lite_to_2 += (end - begin);

        begin = clock();
        image_preprocess.imageRotate(
            lite_dst, tv_out_ratote, (ImageFormat)dstFormat, srcw, srch, 90);
        end = clock();
        lite_to_3 += (end - begin);

        begin = clock();
        image_preprocess.imageFlip(
            lite_dst, tv_out_flip, (ImageFormat)dstFormat, srcw, srch, flip);
        end = clock();
        lite_to_4 += (end - begin);

        clock_t over = clock();
        lite_to += (over - start);

        begin = clock();
        image_preprocess.image2Tensor(lite_dst,
                                      &dst_tensor,
                                      (ImageFormat)dstFormat,
                                      srcw,
                                      srch,
                                      layout,
                                      means,
                                      scales);
        end = clock();
        lite_to_5 += (end - begin);
      }
      to_1 = 1000 * to_1 / CLOCKS_PER_SEC;
      to_2 = 1000 * to_2 / CLOCKS_PER_SEC;
      to_3 = 1000 * to_3 / CLOCKS_PER_SEC;
      to_4 = 1000 * to_4 / CLOCKS_PER_SEC;
      to1 = 1000 * to1 / CLOCKS_PER_SEC;
      std::cout << "opencv convert run time: " << to_1
                << "ms, avg: " << to_1 / test_iter << std::endl;
      std::cout << "opencv resize run time: " << to_2
                << "ms, avg: " << to_2 / test_iter << std::endl;
      std::cout << "opencv rotate run time: " << to_3
                << "ms, avg: " << to_3 / test_iter << std::endl;
      std::cout << "opencv flip  time: " << to_4
                << "ms, avg: " << to_4 / test_iter << std::endl;
      std::cout << "opencv total run time: " << to1
                << "ms, avg: " << to1 / test_iter << std::endl;
      std::cout << "------" << std::endl;

      lite_to_1 = 1000 * lite_to_1 / CLOCKS_PER_SEC;
      lite_to_2 = 1000 * lite_to_2 / CLOCKS_PER_SEC;
      lite_to_3 = 1000 * lite_to_3 / CLOCKS_PER_SEC;
      lite_to_4 = 1000 * lite_to_4 / CLOCKS_PER_SEC;
      lite_to_5 = 1000 * lite_to_5 / CLOCKS_PER_SEC;
      lite_to = 1000 * lite_to / CLOCKS_PER_SEC;
      std::cout << "lite convert run time: " << lite_to_1
                << "ms, avg: " << lite_to_1 / test_iter << std::endl;
      std::cout << "lite resize run time: " << lite_to_2
                << "ms, avg: " << lite_to_2 / test_iter << std::endl;
      std::cout << "lite rotate run time: " << lite_to_3
                << "ms, avg: " << lite_to_3 / test_iter << std::endl;
      std::cout << "lite flip  time: " << lite_to_4
                << "ms, avg: " << lite_to_4 / test_iter << std::endl;
      std::cout << "lite total run time: " << lite_to
                << "ms, avg: " << lite_to / test_iter << std::endl;
      std::cout << "lite img2tensor  time: " << lite_to_5
                << "ms, avg: " << lite_to_5 / test_iter << std::endl;
      std::cout << "------" << std::endl;

      double max_ratio = 0;
      double max_diff = 0;
      const double eps = 1e-6f;
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/resize.jpg";
      std::string convert_name = dst_path + "/convert.jpg";
      std::string rotate_name = dst_path + "/rotate.jpg";
      std::string flip_name = dst_path + "/flip.jpg";
      cv::Mat resize_mat(dsth, dstw, CV_8UC3);
      cv::Mat convert_mat(srch, srcw, CV_8UC3);
      cv::Mat rotate_mat(srch, srcw, CV_8UC3);
      cv::Mat flip_mat(srch, srcw, CV_8UC3);
      fill_with_mat(resize_mat, resize_tmp);
      fill_with_mat(convert_mat, lite_dst);
      printf("lite_dst: %d, %d, %d, %d, %d, %d \n",
             lite_dst[0],
             lite_dst[1],
             lite_dst[2],
             lite_dst[3],
             lite_dst[4],
             lite_dst[5]);
      printf("im_convert: %d, %d, %d, %d, %d, %d \n",
             im_convert.data[0],
             im_convert.data[1],
             im_convert.data[2],
             im_convert.data[3],
             im_convert.data[4],
             im_convert.data[5]);

      fill_with_mat(rotate_mat, tv_out_ratote);
      printf("tv_out_rotate: %d, %d, %d, %d, %d, %d \n",
             tv_out_ratote[0],
             tv_out_ratote[1],
             tv_out_ratote[2],
             tv_out_ratote[3],
             tv_out_ratote[4],
             tv_out_ratote[5]);
      printf("im_rotate: %d, %d, %d, %d, %d, %d \n",
             im_rotate.data[0],
             im_rotate.data[1],
             im_rotate.data[2],
             im_rotate.data[3],
             im_rotate.data[4],
             im_rotate.data[5]);

      fill_with_mat(flip_mat, tv_out_flip);
      printf("tv_out_flip: %d, %d, %d, %d, %d, %d \n",
             tv_out_flip[0],
             tv_out_flip[1],
             tv_out_flip[2],
             tv_out_flip[3],
             tv_out_flip[4],
             tv_out_flip[5]);
      printf("im_flip: %d, %d, %d, %d, %d, %d \n",
             im_flip.data[0],
             im_flip.data[1],
             im_flip.data[2],
             im_flip.data[3],
             im_flip.data[4],
             im_flip.data[5]);
      cv::imwrite(convert_name, convert_mat);
      cv::imwrite(resize_name, resize_mat);
      cv::imwrite(rotate_name, rotate_mat);
      cv::imwrite(flip_name, flip_mat);
      // cv::imwrite(flip_name, im_flip);
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
  int layout = 1;
  std::string model_dir = "mobilenet_v1";
  if (argc > 7) {
    model_dir = argv[7];
  }
  if (argc > 8) {
    flip = atoi(argv[8]);
  }
  if (argc > 9) {
    rotate = atoi(argv[9]);
  }
  if (argc > 10) {
    layout = atoi(argv[10]);
  }
  test_img({3},
           {1, 2, 4},
           image_path,
           dst_path,
           (ImageFormat)srcFormat,
           (ImageFormat)dstFormat,
           width,
           height,
           rotate,
           (FlipParam)flip,
           (LayoutType)layout,
           model_dir,
           20);
  return 0;
}
