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
#include <fstream>
#include <iostream>
#include "../../src/io/paddle_inference_api.h"

using namespace paddle_mobile;        // NOLINT
using namespace paddle_mobile::fpga;  // NOLINT

static const char *g_image = "../images/mobilenet_txtdata/1.txt";
static const char *g_model = "../models/keycurve_l2_regular4_model/__model__";
static const char *g_param =
    "../models/keycurve_l2_regular4_model/model.params";

void readStream(std::string filename, float *buf) {
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }
  int i = 0;
  while (!in.eof()) {
    in >> buf[i];
    i++;
  }
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
  signed char *tmp_data = (signed char *)fpga_malloc(data_size * sizeof(char));
  for (int i = 0; i < data_size; i++) {
    tmp_data[i] = float_to_int8((*data_in)[i] + 128);
  }
  *data_in = (float *)tmp_data;  // NOLINT
  fpga_free(tmp);
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

void dump_stride_float(std::string filename, PaddleTensor input_tensor) {
  auto data_ptr = reinterpret_cast<float *>(input_tensor.data.data());
  int c = (input_tensor.shape)[1];
  int h = (input_tensor.shape)[2];
  int w = (input_tensor.shape)[3];
  int n = (input_tensor.shape)[0];
  float *data_tmp =
      reinterpret_cast<float *>(malloc(c * h * w * sizeof(float)));
  convert_to_chw(&data_ptr, c, h, w, data_tmp);
  std::ofstream out(filename.c_str());
  float result = 0;
  int datasize = abs(c * h * w * n);
  if (datasize == 0) {
    std::cout << "wrong dump data size" << std::endl;
    return;
  }
  for (int i = 0; i < datasize; i++) {
    result = data_tmp[i];
    out << result << std::endl;
  }
  out.close();
}

void dump_stride(std::string filename, PaddleTensor input_tensor) {
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
int main() {
  open_device();
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<paddle_mobile::PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  std::cout << "Finishing loading model" << std::endl;
  int img_length = 256 * 416 * 3;
  auto img = reinterpret_cast<float *>(fpga_malloc(img_length * sizeof(float)));
  readStream(g_image, img);

  std::cout << "Finishing initializing data" << std::endl;
  struct PaddleTensor t_img;
  t_img.dtype = FLOAT32;
  t_img.dtypeid = PaddlekTypeId_t::paddle_float;
  // quantize(&img, img_length);
  // t_img.dtype = INT8;
  // t_img.dtypeid = typeid(int8_t);
  t_img.layout = LAYOUT_HWC;
  t_img.shape = std::vector<int>({1, 256, 416, 3});
  t_img.name = "Image information";
  t_img.data.Reset(img, img_length * sizeof(float));
  // t_img.data.Reset(img, img_length * sizeof(int8_t));
  predictor->FeedPaddleTensors({t_img});

  std::cout << "Finishing feeding data " << std::endl;

  predictor->Predict_From_To(0, -1);
  std::cout << "Finishing predicting " << std::endl;

  std::vector<PaddleTensor> v;        // No need to initialize v
  predictor->FetchPaddleTensors(&v);  // Old data in v will be cleared
  std::cout << "Output number is " << v.size() << std::endl;
  for (int fetchNum = 0; fetchNum < v.size(); fetchNum++) {
    std::string dumpName = "mobilenet_api_fetch_" + std::to_string(fetchNum);
    dump_stride(dumpName, v[fetchNum]);
  }
  return 0;
}
