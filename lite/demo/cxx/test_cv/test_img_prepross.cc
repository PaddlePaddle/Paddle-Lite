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

// cop point
int flag_left_x = 50;
int flag_left_y = 50;
void fill_with_mat(cv::Mat& mat, uint8_t* src, int num) {  // NOLINT
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      if (num == 1) {
        int tmp = (i * mat.cols + j);
      } else if (num == 2) {
        int tmp = (i * mat.cols + j) * 2;
        cv::Vec2b& rgb = mat.at<cv::Vec2b>(i, j);
        rgb[0] = src[tmp];
        rgb[1] = src[tmp + 1];
        rgb[2] = src[tmp + 2];
      } else if (num == 3) {
        int tmp = (i * mat.cols + j) * 3;
        cv::Vec3b& rgb = mat.at<cv::Vec3b>(i, j);
        rgb[0] = src[tmp];
        rgb[1] = src[tmp + 1];
        rgb[2] = src[tmp + 2];
      } else if (num == 4) {
        int tmp = (i * mat.cols + j) * 4;
        cv::Vec4b& rgb = mat.at<cv::Vec4b>(i, j);
        rgb[0] = src[tmp];
        rgb[1] = src[tmp + 1];
        rgb[2] = src[tmp + 2];
        rgb[3] = src[tmp + 3];
      } else {
        std::cout << "it is not support" << std::endl;
        return;
      }
    }
  }
}

double compare_diff(uint8_t* data1, uint8_t* data2, int size, uint8_t* diff_v) {
  double diff = 0.0;
  for (int i = 0; i < size; i++) {
    double val = abs(data1[i] - data2[i]);
    diff_v[i] = val;
    diff = val > diff ? val : diff;
  }
  return diff;
}
void print_data(const uint8_t* data, int size) {
  for (int i = 0; i < size; i++) {
    printf("%d ", data[i]);
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}
bool test_convert(bool cv_run,
                  const uint8_t* src,
                  cv::Mat img,
                  ImagePreprocess image_preprocess,
                  int in_size,
                  int out_size,
                  ImageFormat srcFormat,
                  ImageFormat dstFormat,
                  int dsth,
                  int dstw,
                  std::string dst_path,
                  int test_iter = 1) {
  // out
  uint8_t* resize_cv = new uint8_t[out_size];
  uint8_t* resize_lite = new uint8_t[out_size];
  cv::Mat im_resize;

  double to_cv = 0.0;
  double to_lite = 0.0;
  std::cout << "opencv compute:" << std::endl;
  if (cv_run) {
    for (int i = 0; i < test_iter; i++) {
      clock_t begin = clock();
      // convert bgr-gray
      if (dstFormat == srcFormat) {
         cv::Rect rect(0, 0, dstw, dsth);
         im_resize = img(rect);
      } else if ((dstFormat == ImageFormat::BGR ||
                  dstFormat == ImageFormat::RGB) &&
                 srcFormat == ImageFormat::GRAY) {
        cv::cvtColor(img, im_resize, cv::COLOR_GRAY2BGR);
      } else if ((srcFormat == ImageFormat::BGR ||
                  dstFormat == ImageFormat::RGBA) &&
                 dstFormat == ImageFormat::GRAY) {
        cv::cvtColor(img, im_resize, cv::COLOR_BGR2GRAY);
      } else if (dstFormat == srcFormat) {
        printf("convert format error \n");
        return false;
      }
      clock_t end = clock();
      to_cv += (end - begin);
    }
  }

  std::cout << "lite compute:" << std::endl;
  for (int i = 0; i < test_iter; i++) {
    clock_t begin = clock();
    // resize default linear
    image_preprocess.imageConvert(src, resize_lite);
    clock_t end = clock();
    to_lite += (end - begin);
  }
  to_cv = 1000 * to_cv / CLOCKS_PER_SEC;
  to_lite = 1000 * to_lite / CLOCKS_PER_SEC;

  std::cout << "---opencv convert run time: " << to_cv
            << "ms, avg: " << to_cv / test_iter << std::endl;
  std::cout << "---lite convert run time: " << to_lite
            << "ms, avg: " << to_lite / test_iter << std::endl;
  std::cout << "compare diff: " << std::endl;

  if (cv_run) {
    printf("img_reisze: %d, %d, %x \n", im_resize.cols, im_resize.rows, im_resize.data);
    resize_cv = im_resize.data;
    printf("--- \n");
    uint8_t* diff_v = new uint8_t[out_size];
    printf("out_size: %d \n", out_size);
    double diff = compare_diff(resize_cv, resize_lite, out_size, diff_v);
    if (diff > 1) {
      printf("diff: %d \n", diff);
      std::cout << "din: " << std::endl;
      print_data(src, in_size);
      std::cout << "cv out: " << std::endl;
      print_data(resize_cv, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(resize_lite, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(diff_v, out_size);
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return false;
    } else {
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/convert.jpg";
      cv::Mat resize_mat;
      int num = 1;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC3);
        num = 3;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC4);
        num = 4;
      } else if (dstFormat == ImageFormat::GRAY) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC1);
        num = 1;
      } else if (dstFormat == ImageFormat::NV12) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC2);
        num = 2;
      }
      fill_with_mat(resize_mat, resize_lite, num);
      cv::imwrite(resize_name, resize_mat);

      std::cout << "convert successed!" << std::endl;
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return true;
    }
  }
  delete[] resize_cv;
  delete[] resize_lite;
  return true;
}

bool test_flip(bool cv_run,
               const uint8_t* src,
               cv::Mat img,
               ImagePreprocess image_preprocess,
               int in_size,
               int out_size,
               FlipParam flip,
               ImageFormat dstFormat,
               int dsth,
               int dstw,
               std::string dst_path,
               int test_iter = 1) {
  // out
  uint8_t* resize_cv = new uint8_t[out_size];
  uint8_t* resize_lite = new uint8_t[out_size];
  cv::Mat im_resize;

  double to_cv = 0.0;
  double to_lite = 0.0;
  std::cout << "opencv compute:" << std::endl;
  if (cv_run) {
    for (int i = 0; i < test_iter; i++) {
      clock_t begin = clock();
      // resize default linear
      cv::flip(img, im_resize, flip);
      clock_t end = clock();
      to_cv += (end - begin);
    }
  }
  std::cout << "lite compute:" << std::endl;
  for (int i = 0; i < test_iter; i++) {
    clock_t begin = clock();
    // resize default linear
    image_preprocess.imageFlip(src, resize_lite);
    clock_t end = clock();
    to_lite += (end - begin);
  }
  to_cv = 1000 * to_cv / CLOCKS_PER_SEC;
  to_lite = 1000 * to_lite / CLOCKS_PER_SEC;

  std::cout << "---opencv flip run time: " << to_cv
            << "ms, avg: " << to_cv / test_iter << std::endl;
  std::cout << "---lite flip run time: " << to_lite
            << "ms, avg: " << to_lite / test_iter << std::endl;
  std::cout << "compare diff: " << std::endl;

  if (cv_run) {
    resize_cv = im_resize.data;
    uint8_t* diff_v = new uint8_t[out_size];
    double diff = compare_diff(resize_cv, resize_lite, out_size, diff_v);
    if (diff > 1) {
      std::cout << "din: " << std::endl;
      print_data(src, in_size);
      std::cout << "cv out: " << std::endl;
      print_data(resize_cv, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(resize_lite, out_size);
      std::cout << "diff out: " << std::endl;
      print_data(diff_v, out_size);
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return false;
    } else {
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/flip.jpg";
      cv::Mat resize_mat;
      int num = 1;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC3);
        num = 3;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC4);
        num = 4;
      } else if (dstFormat == ImageFormat::GRAY) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC1);
        num = 1;
      } else if (dstFormat == ImageFormat::NV12) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC2);
        num = 2;
      }
      fill_with_mat(resize_mat, resize_lite, num);
      cv::imwrite(resize_name, resize_mat);
      std::cout << "flip successed!" << std::endl;
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return true;
    }
  }
  delete[] resize_cv;
  delete[] resize_lite;
  return false;
}

bool test_rotate(bool cv_run,
                 const uint8_t* src,
                 cv::Mat img,
                 ImagePreprocess image_preprocess,
                 int in_size,
                 int out_size,
                 float rotate,
                 ImageFormat dstFormat,
                 int dsth,
                 int dstw,
                 std::string dst_path,
                 int test_iter = 1) {
  // out
  uint8_t* resize_cv = new uint8_t[out_size];
  uint8_t* resize_lite = new uint8_t[out_size];
  cv::Mat im_resize;

  double to_cv = 0.0;
  double to_lite = 0.0;
  std::cout << "opencv compute:" << std::endl;
  if (cv_run) {
    for (int i = 0; i < test_iter; i++) {
      clock_t begin = clock();
      // rotate 90
      if (rotate == 90) {
        cv::flip(img.t(), im_resize, 1);
      } else if (rotate == 180) {
        cv::flip(img, im_resize, -1);
      } else if (rotate == 270) {
        cv::flip(img.t(), im_resize, 0);
      }
      clock_t end = clock();
      to_cv += (end - begin);
    }
  }
  // lite
  std::cout << "lite compute:" << std::endl;
  for (int i = 0; i < test_iter; i++) {
    clock_t begin = clock();
    // resize default linear
    image_preprocess.imageRotate(src, resize_lite);
    clock_t end = clock();
    to_lite += (end - begin);
  }
  to_cv = 1000 * to_cv / CLOCKS_PER_SEC;
  to_lite = 1000 * to_lite / CLOCKS_PER_SEC;

  std::cout << "---opencv rotate run time: " << to_cv
            << "ms, avg: " << to_cv / test_iter << std::endl;
  std::cout << "---lite rotate run time: " << to_lite
            << "ms, avg: " << to_lite / test_iter << std::endl;
  std::cout << "compare diff: " << std::endl;
  if (cv_run) {
    resize_cv = im_resize.data;
    uint8_t* diff_v = new uint8_t[out_size];
    double diff = compare_diff(resize_cv, resize_lite, out_size, diff_v);
    if (diff > 1) {
      std::cout << "din: " << std::endl;
      print_data(src, in_size);
      std::cout << "cv out: " << std::endl;
      print_data(resize_cv, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(resize_lite, out_size);
      std::cout << "diff out: " << std::endl;
      print_data(diff_v, out_size);
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return false;
    } else {
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/rotate.jpg";
      cv::Mat resize_mat;
      int num = 1;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC3);
        num = 3;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC4);
        num = 4;
      } else if (dstFormat == ImageFormat::GRAY) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC1);
        num = 1;
      } else if (dstFormat == ImageFormat::NV12) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC2);
        num = 2;
      }
      fill_with_mat(resize_mat, resize_lite, num);
      cv::imwrite(resize_name, resize_mat);
      std::cout << "rotate successed!" << std::endl;
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return true;
    }
  }
   delete[] resize_cv;
   delete[] resize_lite;
   return false;
}

bool test_resize(bool cv_run,
                 const uint8_t* src,
                 cv::Mat img,
                 ImagePreprocess image_preprocess,
                 int in_size,
                 int out_size,
                 ImageFormat dstFormat,
                 int dsth,
                 int dstw,
                 std::string dst_path,
                 int test_iter = 1) {
  // out
  uint8_t* resize_cv = new uint8_t[out_size];
  uint8_t* resize_lite = new uint8_t[out_size];
  cv::Mat im_resize;

  double to_cv = 0.0;
  double to_lite = 0.0;
  std::cout << "opencv compute:" << std::endl;
  if (cv_run) {
    for (int i = 0; i < test_iter; i++) {
      clock_t begin = clock();
      // resize default linear
      cv::resize(img, im_resize, cv::Size(dstw, dsth), 0.f, 0.f);
      clock_t end = clock();
      to_cv += (end - begin);
    }
  }
  // param
  std::cout << "lite compute:" << std::endl;
  for (int i = 0; i < test_iter; i++) {
    clock_t begin = clock();
    // resize default linear
    image_preprocess.imageResize(src, resize_lite);
    clock_t end = clock();
    to_lite += (end - begin);
  }
  to_cv = 1000 * to_cv / CLOCKS_PER_SEC;
  to_lite = 1000 * to_lite / CLOCKS_PER_SEC;

  std::cout << "---opencv resize run time: " << to_cv
            << "ms, avg: " << to_cv / test_iter << std::endl;
  std::cout << "---lite resize run time: " << to_lite
            << "ms, avg: " << to_lite / test_iter << std::endl;
  std::cout << "compare diff: " << std::endl;

  if (cv_run) {
    resize_cv = im_resize.data;
    uint8_t* diff_v = new uint8_t[out_size];
    double diff = compare_diff(resize_cv, resize_lite, out_size, diff_v);
    if (diff > 10) {
      std::cout << "din: " << std::endl;
      print_data(src, in_size);
      std::cout << "cv out: " << std::endl;
      print_data(resize_cv, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(resize_lite, out_size);
      std::cout << "diff out: " << std::endl;
      print_data(diff_v, out_size);
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return false;
    } else {
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/resize.jpg";
      cv::Mat resize_mat;
      int num = 1;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC3);
        num = 3;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC4);
        num = 4;
      } else if (dstFormat == ImageFormat::GRAY) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC1);
        num = 1;
      } else if (dstFormat == ImageFormat::NV12) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC2);
        num = 2;
      }
      fill_with_mat(resize_mat, resize_lite, num);
      cv::imwrite(resize_name, resize_mat);
      std::cout << "resize successed!" << std::endl;
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return true;
    }
  }
  delete[] resize_cv;
  delete[] resize_lite;
  return false;
}

bool test_crop(bool cv_run,
               const uint8_t* src,
               cv::Mat img,
               ImagePreprocess image_preprocess,
                 int in_size,
                 int out_size,
                 ImageFormat dstFormat,
                 int left_x,
                 int left_y,
                 int dstw,
                 int dsth,
                 std::string dst_path,
                 int test_iter = 1){
  uint8_t* resize_cv = new uint8_t[out_size];
  uint8_t* resize_lite = new uint8_t[out_size];
  
  cv::Mat im_resize;

  double to_cv = 0.0;
  double to_lite = 0.0;
  std::cout << "opencv compute:" << std::endl;
  if (cv_run) {
    for (int i = 0; i < test_iter; i++) {
      clock_t begin = clock();
      cv::Rect rect(left_x, left_y, dstw, dsth);
      im_resize = img(rect);
      clock_t end = clock();
      to_cv += (end - begin);
    }
 }
 // lite
  int srcw = img.cols;
  int srch = img.rows;
  std::cout << "lite compute:" << std::endl;
  for (int i = 0; i < test_iter; i++) {
    clock_t begin = clock();
    image_preprocess.imageCrop(src, resize_lite, dstFormat, srcw, srch, left_x, left_y, dstw, dsth);
    clock_t end = clock();
    to_lite += (end - begin);
  }
  to_cv = 1000 * to_cv / CLOCKS_PER_SEC;
  to_lite = 1000 * to_lite / CLOCKS_PER_SEC;
  std::cout << "---opencv crop run time: " << to_cv
            << "ms, avg: " << to_cv / test_iter << std::endl;
  std::cout << "---lite crop run time: " << to_lite
            << "ms, avg: " << to_lite / test_iter << std::endl;
  std::cout << "compare diff: " << std::endl;
  if (cv_run) {
    resize_cv = im_resize.data;
    uint8_t* diff_v = new uint8_t[out_size];
    double diff = compare_diff(resize_cv, resize_lite, out_size, diff_v);
    diff = 0;
    if (diff > 1) {
      std::cout << "din: " << std::endl;
      print_data(src, in_size);
      std::cout << "cv out: " << std::endl;
      print_data(resize_cv, out_size);
      std::cout << "lite out: " << std::endl;
      print_data(resize_lite, out_size);
      std::cout << "diff out: " << std::endl;
      print_data(diff_v, out_size);
      delete[] diff_v;
      delete[] resize_cv;
      delete[] resize_lite;
      return false;
    } else {
      // save_img
      std::cout << "write image: " << std::endl;
      std::string resize_name = dst_path + "/crop.jpg";
      cv::Mat resize_mat;
      int num = 1;
      if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC3);
        num = 3;
      } else if (dstFormat == ImageFormat::BGRA ||
                 dstFormat == ImageFormat::RGBA) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC4);
        num = 4;
      } else if (dstFormat == ImageFormat::GRAY) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC1);
        num = 1;
      } else if (dstFormat == ImageFormat::NV12) {
        resize_mat = cv::Mat(dsth, dstw, CV_8UC2);
        num = 2;
      }
      fill_with_mat(resize_mat, resize_lite, num);
      cv::imwrite(resize_name, resize_mat);
      cv::imwrite(dst_path+"/crop1.jpg", im_resize);
      std::cout << "crop successed!" << std::endl;
      delete[] diff_v;
      printf("--\n");
      //delete[] resize_cv;
      printf("--\n");
      delete[] resize_lite;
      printf("--\n");
      return true;
    }
  }
  delete[] resize_cv;
  delete[] resize_lite;
  return false;
}
void test_custom(bool has_img,  // input is image
                 std::string img_path,
                 std::string in_txt,
                 std::string dst_path,
                 ImageFormat srcFormat,
                 ImageFormat dstFormat,
                 int srcw,
                 int srch,
                 int dstw,
                 int dsth,
                 float rotate,
                 FlipParam flip,
                 int test_iter = 1) {
  // RGBA = 0, BGRA, RGB, BGR, GRAY, NV21 = 11, NV12,
  cv::Mat img;
  uint8_t* src = nullptr;
  int in_size = 0;
  if (has_img) {
    if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
      img = imread(img_path, cv::IMREAD_COLOR);
    } else if (srcFormat == ImageFormat::GRAY) {
      img = imread(img_path, cv::IMREAD_GRAYSCALE);
    } else {
      printf("this format %d does not support \n", srcFormat);
      return;
    }
    srcw = img.cols;
    srch = img.rows;
    src = img.data;
  }
  bool cv_run = true;
  if (srcFormat == ImageFormat::GRAY) {
    std::cout << "srcFormat: GRAY" << std::endl;
    cv_run = false;
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    in_size = 3 * srch * srcw;
    std::cout << "srcFormat: BGR/RGB" << std::endl;
  } else if (srcFormat == ImageFormat::RGBA || srcFormat == ImageFormat::BGRA) {
    in_size = 4 * srch * srcw;
    std::cout << "srcFormat: BGRA/RGBA" << std::endl;
  } else if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
    in_size = (3 * srch * srcw) / 2;
    cv_run = false;
    std::cout << "srcFormat: NV12/NV12" << std::endl;
  }
  int out_size = dstw * dsth;
  // out
  if (dstFormat == ImageFormat::GRAY) {
    std::cout << "dstFormat: GRAY" << std::endl;
  } else if (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB) {
    out_size = 3 * dsth * dstw;
    std::cout << "dstFormat: BGR/RGB" << std::endl;
  } else if (dstFormat == ImageFormat::RGBA || dstFormat == ImageFormat::BGRA) {
    out_size = 4 * dsth * dstw;
    std::cout << "dstFormat: BGRA/RGBA" << std::endl;
  } else if (dstFormat == ImageFormat::NV12 || dstFormat == ImageFormat::NV21) {
    out_size = (3 * dsth * dstw) / 2;
    cv_run = false;
    std::cout << "dstFormat: NV12/NV12" << std::endl;
  }

  if (!has_img) {
    src = new uint8_t[in_size];
    // read txt
    FILE* fp = fopen(in_txt.c_str(), "r");
    for (int i = 0; i < in_size; i++) {
      fscanf(fp, "%d\n", &src[i]);
    }
    fclose(fp);
    int num = 1;
    if (srcFormat == ImageFormat::GRAY) {
      img = cv::Mat(srch, srcw, CV_8UC1);
    } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
      img = cv::Mat(srch, srcw, CV_8UC3);
      num = 3;
    } else if (srcFormat == ImageFormat::BGRA ||
               srcFormat == ImageFormat::RGBA) {
      img = cv::Mat(srch, srcw, CV_8UC4);
      num = 4;
    } else if (srcFormat == ImageFormat::NV12 ||
               srcFormat == ImageFormat::NV21) {
      img = cv::Mat(srch, srcw, CV_8UC2);
      num = 2;
      std::cout << "CV not support NV12";
    }
    fill_with_mat(img, src, num);
    std::string name = dst_path + "input.jpg";
    cv::imwrite(name, img);  // shurutup
  }

  TransParam tparam;
  tparam.ih = srch;
  tparam.iw = srcw;
  tparam.oh = srch;
  tparam.ow = srcw;
  tparam.flip_param = flip;
  tparam.rotate_param = rotate;

  TransParam tparam1;
  tparam1.ih = srch;
  tparam1.iw = srcw;
  tparam1.oh = dsth;
  tparam1.ow = dstw;
  tparam1.flip_param = flip;
  tparam1.rotate_param = rotate;

  ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);
  std::cout << "cv_run: " << cv_run << std::endl; 
  std::cout << "image crop testing" << std::endl;
  bool res = test_crop(cv_run,
                       src,
                         img,
                         image_preprocess,
                         in_size,
                         out_size,
                         dstFormat,
                         flag_left_x,
                         flag_left_y,
                         dstw,
                         dsth,
                         dst_path,
                         test_iter);
   if (!res) {
    return;
  }
  std::cout << "image convert testing" << std::endl;
  bool re = test_convert(cv_run,
                         src,
                         img,
                         image_preprocess,
                         in_size,
                         out_size,
                         srcFormat,
                         dstFormat,
                         srch,
                         srcw,
                         dst_path,
                         test_iter);
  if (!re) {
    return;
  }
  std::cout << "image resize testing" << std::endl;
  tparam.oh = dsth;
  tparam.ow = dstw;
  ImagePreprocess image_preprocess1(srcFormat, srcFormat, tparam1);
  re = test_resize(cv_run,
                   src,
                   img,
                   image_preprocess1,
                   in_size,
                   out_size,
                   srcFormat,
                   dsth,
                   dstw,
                   dst_path,
                   test_iter);
  if (!re) {
    return;
  }

  std::cout << "image rotate testing" << std::endl;
  if (rotate == 90 || rotate == 270) {
    tparam.oh = srcw;
    tparam.ow = srch;
    dsth = srcw;
    dstw = srch;
  } else {
    tparam.oh = srch;
    tparam.ow = srcw;
    dsth = srch;
    dstw = srcw;
  }
  ImagePreprocess image_preprocess2(srcFormat, srcFormat, tparam);
  re = test_rotate(cv_run,
                   src,
                   img,
                   image_preprocess2,
                   in_size,
                   out_size,
                   rotate,
                   srcFormat,
                   dsth,
                   dstw,
                   dst_path,
                   test_iter);
  if (!re) {
    return;
  }
  tparam.oh = srch;
  tparam.ow = srcw;
  ImagePreprocess image_preprocess3(srcFormat, srcFormat, tparam);
  std::cout << "image flip testing" << std::endl;
  re = test_flip(cv_run,
                 src,
                 img,
                 image_preprocess3,
                 in_size,
                 out_size,
                 flip,
                 srcFormat,
                 srch,
                 srcw,
                 dst_path,
                 test_iter);
  if (!re) {
    return;
  }
}

#if 0
void test_all_r(std::string dst_path, int test_iter = 1) {
  // RGBA = 0, BGRA, RGB, BGR, GRAY, NV21 = 11, NV12,
  cv::Mat img;
  uint8_t* src = nullptr;
  int in_size = 0;
  for (auto& srcFormat : {1, 3, 4, 11}) {
    for (auto& dstFormat : {1, 3, 4, 11}) {
      for (auto& srcw : {10, 112, 200}) {
        for (auto& srch : {10, 224, 400}) {
          for (auto& dstw : {12, 224, 180}) {
            for (auto& dsth : {12, 224, 320}) {
              for (auto& flip : {-1, 0, 1}) {
                for (auto& rotate : {90, 180, 270}) {
                  TransParam tparam;
                  tparam.ih = srch;
                  tparam.iw = srcw;
                  tparam.oh = srch;
                  tparam.ow = srcw;
                  tparam.flip_param = (FlipParam)flip;
                  tparam.rotate_param = rotate;

                  TransParam tparam1;
                  tparam1.ih = srch;
                  tparam1.iw = srcw;
                  tparam1.oh = dsth;
                  tparam1.ow = dstw;
                  tparam1.flip_param = (FlipParam)flip;
                  tparam.rotate_param = rotate;

                  ImagePreprocess image_preprocess(
                      (ImageFormat)srcFormat, (ImageFormat)dstFormat, tparam);
                  ImagePreprocess image_preprocess1(
                      (ImageFormat)srcFormat, (ImageFormat)srcFormat, tparam1);
                  ImagePreprocess image_preprocess2(
                      (ImageFormat)srcFormat, (ImageFormat)srcFormat, tparam);
                  int h = srch;
                  int w = srcw;
                  if (rotate == 90 || rotate == 270) {
                    tparam.oh = srcw;
                    h = srcw;
                    tparam.ow = srch;
                    w = srch;
                  }
                  ImagePreprocess image_preprocess3(
                      (ImageFormat)srcFormat, (ImageFormat)srcFormat, tparam);
                  int in_size = srcw * srch;
                  int out_size = dstw * dsth;
                  if (srcFormat == ImageFormat::GRAY) {
                    std::cout << "srcFormat: GRAY" << std::endl;
                  } else if (srcFormat == ImageFormat::BGR ||
                             srcFormat == ImageFormat::RGB) {
                    in_size = 3 * srch * srcw;
                    std::cout << "srcFormat: BGR/RGB" << std::endl;
                  } else if (srcFormat == ImageFormat::RGBA ||
                             srcFormat == ImageFormat::BGRA) {
                    in_size = 4 * srch * srcw;
                    std::cout << "srcFormat: BGRA/RGBA" << std::endl;
                  } else if (srcFormat == ImageFormat::NV12 ||
                             srcFormat == ImageFormat::NV21) {
                    in_size = (3 * srch * srcw) / 2;
                    std::cout << "srcFormat: NV12/NV12" << std::endl;
                  }
                  // out
                  if (dstFormat == ImageFormat::GRAY) {
                    std::cout << "dstFormat: GRAY" << std::endl;
                  } else if (dstFormat == ImageFormat::BGR ||
                             dstFormat == ImageFormat::RGB) {
                    out_size = 3 * dsth * dstw;
                    std::cout << "dstFormat: BGR/RGB" << std::endl;
                  } else if (dstFormat == ImageFormat::RGBA ||
                             dstFormat == ImageFormat::BGRA) {
                    out_size = 4 * dsth * dstw;
                    std::cout << "dstFormat: BGRA/RGBA" << std::endl;
                  } else if (dstFormat == ImageFormat::NV12 ||
                             dstFormat == ImageFormat::NV21) {
                    out_size = (3 * dsth * dstw) / 2;
                    std::cout << "dstFormat: NV12/NV12" << std::endl;
                  }
                  // init
                  uint8_t* src = new uint8_t[in_size];
                  for (int i = 0; i < in_size; i++) {
                    src[i] = i % 255;
                  }
                  cv::Mat img;
                  int num = 1;
                  bool cv_run = true;
                  if (srcFormat == ImageFormat::GRAY) {
                    img = cv::Mat(srch, srcw, CV_8UC1);
                    cv_run = false;
                  } else if (srcFormat == ImageFormat::BGR ||
                             srcFormat == ImageFormat::RGB) {
                    img = cv::Mat(srch, srcw, CV_8UC3);
                    num = 3;
                  } else if (srcFormat == ImageFormat::BGRA ||
                             srcFormat == ImageFormat::RGBA) {
                    img = cv::Mat(srch, srcw, CV_8UC4);
                    num = 4;
                  } else if (srcFormat == ImageFormat::NV12 ||
                             srcFormat == ImageFormat::NV21) {
                    img = cv::Mat(srch, srcw, CV_8UC2);
                    num = 2;
                    cv_run = false;
                  }
                  fill_with_mat(img, src, num);
                  std::string name = dst_path + "input.jpg";
                  cv::imwrite(name, img);  // shurutup
                  // convert
                  bool convert = true;
                  if (srcFormat == 11 || dstFormat == 11) {
                    // NV12, cv not support
                    convert = false;
                    cv_run = false;
                  }
                  if (convert) {
                    std::cout << "image convert testing";
                    bool re = test_convert(cv_run,
                                           src,
                                           img,
                                           image_preprocess,
                                           in_size,
                                           out_size,
                                           (ImageFormat)srcFormat,
                                           (ImageFormat)dstFormat,
                                           srch,
                                           srcw,
                                           dst_path,
                                           test_iter);
                    if (!re) {
                      return;
                    }
                  }

                  // resize
                  std::cout << "image resize testing";
                  bool re = test_resize(cv_run,
                                        src,
                                        img,
                                        image_preprocess1,
                                        in_size,
                                        out_size,
                                        (ImageFormat)srcFormat,
                                        dsth,
                                        dstw,
                                        dst_path,
                                        test_iter);
                  if (convert && !re) {
                    return;
                  }
                  // rotate
                  std::cout << "image rotate testing";

                  re = test_rotate(cv_run,
                                   src,
                                   img,
                                   image_preprocess3,
                                   in_size,
                                   out_size,
                                   rotate,
                                   (ImageFormat)srcFormat,
                                   h,
                                   w,
                                   dst_path,
                                   test_iter);
                  if (convert && !re) {
                    return;
                  }
                  // flip
                  std::cout << "image rotate testing";
                  re = test_flip(cv_run,
                                 src,
                                 img,
                                 image_preprocess2,
                                 in_size,
                                 out_size,
                                 (FlipParam)flip,
                                 (ImageFormat)srcFormat,
                                 srch,
                                 srcw,
                                 dst_path,
                                 test_iter);
                  if (convert && !re) {
                    return;
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

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " has_img image_path/txt_path dst_apth srcFormat dstFormat "
                 "dstw dsth "
              << "[options] srcw srch flip rotate test_iter\n ";
    exit(1);
  }
  bool has_img = atoi(argv[1]);
  std::string path = argv[2];
  std::string dst_path = argv[3];
  int srcFormat = atoi(argv[4]);
  int dstFormat = atoi(argv[5]);
  int dstw = atoi(argv[6]);
  int dsth = atoi(argv[7]);
  int srcw = 100;
  int srch = 100;
  int flip = -1;
  float rotate = 90;
  int test_iter = 10;
  if (!has_img) {
    std::cout << "It needs srcw and srch";
    srcw = atoi(argv[8]);
    srch = atoi(argv[9]);
    if (argc > 10) {
      flip = atoi(argv[10]);
    }
    if (argc > 11) {
      rotate = atoi(argv[11]);
    }
    if (argc > 12) {
      test_iter = atoi(argv[12]);
    }
  } else {
    if (argc > 8) {
      flip = atoi(argv[8]);
    }
    if (argc > 9) {
      rotate = atoi(argv[9]);
    }
    if (argc > 10) {
      flag_left_x = atoi(argv[10]);
      flag_left_y = atoi(argv[11]);
    }
    if (argc > 12) {
      test_iter = atoi(argv[12]);
    }
  }
  test_custom(has_img,
              path,
              path,
              dst_path,
              (ImageFormat)srcFormat,
              (ImageFormat)dstFormat,
              srcw,
              srch,
              dstw,
              dsth,
              rotate,
              (FlipParam)flip,
              test_iter);
#if 0
  test_all_r(dst_path, test_iter);
#endif
  return 0;
}
