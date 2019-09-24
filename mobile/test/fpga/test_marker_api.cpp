/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_MOBILE_FPGA
#define PADDLE_MOBILE_FPGA
#endif
#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../../src/io/paddle_inference_api.h"

using namespace paddle_mobile;        // NOLINT
using namespace paddle_mobile::fpga;  // NOLINT

static const char *g_image = "../models/marker/model/image.bin";
static const char *g_model = "../models/marker/model/model";
static const char *g_param = "../models/marker/model/params";

static const char *g_image1 = "../models/marker2/model/marker.bin";
static const char *g_model1 = "../models/marker2/model/model";
static const char *g_param1 = "../models/marker2/model/params";

void readStream(std::string filename, char *buf) {
  std::ifstream in;
  in.open(filename, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }

  in.seekg(0, std::ios::end);  // go to the end
  auto length = in.tellg();    // report location (this is the length)
  in.seekg(0, std::ios::beg);  // go back to the beginning
  in.read(buf, length);
  in.close();
}
signed char float_to_int8(float fdata) {
  if (fdata < 0.0) {
    fdata -= 0.5;
  } else {
    fdata += 0.5;
  }
  return (signed char)fdata;
}
void quantize(float **data_in, int data_size) {
  float *tmp = *data_in;
  signed char *tmp_data =
      (signed char *)paddle_mobile::fpga::fpga_malloc(data_size * sizeof(char));
  for (int i = 0; i < data_size; i++) {
    tmp_data[i] = float_to_int8((*data_in)[i] + 128);
  }
  *data_in = (float *)tmp_data;  // NOLINT
  paddle_mobile::fpga::fpga_free(tmp);
}

void convert_to_chw(float **data_in, int channel, int height, int width,
                    float *data_tmp) {
  int64_t amount_per_side = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channel; c++) {
        *(data_tmp + c * amount_per_side + width * h + w) = *((*data_in)++);
      }
    }
  }
}

void dump_stride_float(std::string filename,
                       paddle_mobile::PaddleTensor input_tensor) {
  auto data_ptr = reinterpret_cast<float *>(input_tensor.data.data());
  int c = (input_tensor.shape)[1];
  int h = (input_tensor.shape)[2];
  int w = (input_tensor.shape)[3];
  int n = (input_tensor.shape)[0];
  float *data_tmp =
      reinterpret_cast<float *>(malloc(c * h * w * sizeof(float)));
  // convert_to_chw(&data_ptr, c, h, w, data_tmp);
  std::ofstream out(filename.c_str());
  float result = 0;
  int datasize = abs(c * h * w * n);
  if (datasize == 0) {
    std::cout << "wrong dump data size" << std::endl;
    return;
  }
  for (int i = 0; i < datasize; i++) {
    result = data_ptr[i];
    out << result << std::endl;
  }
  out.close();
}

void dump_stride(std::string filename,
                 paddle_mobile::PaddleTensor input_tensor) {
  if (input_tensor.dtypeid == PaddlekTypeId_t::paddle_float) {
    dump_stride_float(filename, input_tensor);
  } else {
    std::cout << "only support dumping float data" << std::endl;
  }
}
PaddleMobileConfig GetConfig() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  config.prog_file = g_model;
  config.param_file = g_param;
  config.thread_num = 1;
  config.batch_size = 1;
  config.optimize = true;
  config.lod_mode = true;
  config.quantification = false;
  return config;
}
PaddleMobileConfig GetConfig1() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  config.prog_file = g_model1;
  config.param_file = g_param1;
  config.thread_num = 1;
  config.batch_size = 1;
  config.optimize = true;
  config.lod_mode = true;
  config.quantification = false;
  return config;
}

int main() {
  open_device();
  timeval start11, end11;
  long dif_sec, dif_usec;  // NOLINT

  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  std::cout << "Finishing loading model" << std::endl;

  float img_info[3] = {432, 1280, 1.0f};
  int img_length = 432 * 1280 * 3;
  auto img = reinterpret_cast<float *>(fpga_malloc(img_length * sizeof(float)));
  readStream(g_image, reinterpret_cast<char *>(img));

  std::cout << "Finishing initializing data" << std::endl;
  struct PaddleTensor t_img_info, t_img;
  t_img_info.dtypeid = PaddlekTypeId_t::paddle_float;
  t_img_info.layout = LAYOUT_HWC;
  t_img_info.shape = std::vector<int>({1, 3});
  t_img_info.name = "Image information";
  t_img_info.data.Reset(img_info, 3 * sizeof(float));

  t_img.dtypeid = PaddlekTypeId_t::paddle_float;
  // quantize(&img, img_length);
  // t_img.dtypeid = typeid(int8_t);
  t_img.layout = LAYOUT_HWC;
  t_img.shape = std::vector<int>({1, 432, 1280, 3});
  t_img.name = "Image information";
  t_img.data.Reset(img, img_length * sizeof(float));
  // t_img.data.Reset(img, img_length * sizeof(int8_t));
  // for(int i = 0; i < 100; ++i){
  predictor->FeedPaddleTensors({t_img_info, t_img});

  std::cout << "Finishing feeding data " << std::endl;

  gettimeofday(&start11, NULL);
  predictor->Predict_From_To(0, -1);
  gettimeofday(&end11, NULL);
  dif_sec = end11.tv_sec - start11.tv_sec;
  dif_usec = end11.tv_usec - start11.tv_usec;
  std::cout << "marker1 total"
            << " cost time: " << (dif_sec * 1000000 + dif_usec) << "  us"
            << std::endl;
  std::cout << "Finishing predicting " << std::endl;

  std::vector<paddle_mobile::PaddleTensor> v;  // No need to initialize v
  predictor->FetchPaddleTensors(&v);           // Old data in v will be cleared
  std::cout << "Output number is " << v.size() << std::endl;
  for (int fetchNum = 0; fetchNum < v.size(); fetchNum++) {
    std::string dumpName = "marker_api_fetch_" + std::to_string(fetchNum);
    // dump_stride(dumpName, v[fetchNum]);
  }
  fpga_free(img);

  PaddleMobileConfig config1 = GetConfig1();
  auto predictor1 =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config1);

  std::cout << "Finishing loading model" << std::endl;
  for (int i = 0; i < 1; ++i) {
    int img_length1 = 144 * 14 * 14;
    auto img1 =
        reinterpret_cast<float *>(fpga_malloc(img_length1 * sizeof(float)));
    readStream(g_image1, reinterpret_cast<char *>(img1));

    std::cout << "Finishing initializing data" << std::endl;
    struct PaddleTensor t_img1;

    t_img1.dtypeid = PaddlekTypeId_t::paddle_float;
    t_img1.layout = LAYOUT_HWC;
    t_img1.shape = std::vector<int>({1, 14, 14, 144});
    t_img1.name = "Image information";
    t_img1.data.Reset(img1, img_length1 * sizeof(float));
    predictor1->FeedPaddleTensors({t_img1});

    std::cout << "Finishing feeding data " << std::endl;

    gettimeofday(&start11, NULL);
    predictor1->Predict_From_To(0, -1);
    gettimeofday(&end11, NULL);
    dif_sec = end11.tv_sec - start11.tv_sec;
    dif_usec = end11.tv_usec - start11.tv_usec;
    std::cout << "marker2 total"
              << "    cost time: " << (dif_sec * 1000000 + dif_usec) << "  us"
              << std::endl;
    std::cout << "Finishing predicting " << std::endl;

    std::vector<paddle_mobile::PaddleTensor> v1;  // No need to initialize v
    predictor1->FetchPaddleTensors(&v1);  // Old data in v will be cleared
    std::cout << "Output number is " << v1.size() << std::endl;
    for (int fetchNum = 0; fetchNum < v1.size(); fetchNum++) {
      std::string dumpName = "marker2_api_fetch_" + std::to_string(fetchNum);
      dump_stride(dumpName, v1[fetchNum]);
    }
    fpga_free(img1);
  }
  return 0;
}
