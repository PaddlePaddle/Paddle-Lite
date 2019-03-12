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
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../test_include.h"

#ifdef PADDLE_MOBILE_FPGA_V1
#include "fpga/V1/api.h"
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#endif

void readStream(std::string filename, float *buf) {
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }
  string strOne;
  int i = 0;
  while (!in.eof()) {
    in >> buf[i];
    i++;
  }
  in.close();
}

void convert_to_chw(int16_t **data_in, int channel, int height, int width,
                    int16_t *data_tmp) {
  int64_t amount_per_side = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channel; c++) {
        *(data_tmp + c * amount_per_side + width * h + w) = *((*data_in)++);
      }
    }
  }
}

void dump(std::string filename, Tensor input_tensor) {
  auto dataptr = reinterpret_cast<half *>(input_tensor.get_data());
  std::ofstream out(filename.c_str());
  float result = 0;
  for (int i = 0; i < input_tensor.numel(); ++i) {
    result = paddle_mobile::fpga::fp16_2_fp32(dataptr[i]);
    out << result << std::endl;
  }
  out.close();
}
void dump_stride_half(std::string filename, Tensor input_tensor,
                      const int dumpnum) {
  int c = (input_tensor.dims())[1];
  int h = (input_tensor.dims())[2];
  int w = (input_tensor.dims())[3];
  auto data_ptr = input_tensor.get_data();
  auto *data_tmp =
      reinterpret_cast<half *>(malloc(c * h * w * sizeof(int16_t)));
  auto *data_ptr_16 = reinterpret_cast<half *>(data_ptr);
  convert_to_chw(&data_ptr_16, c, h, w, data_tmp);
  std::ofstream out(filename.c_str());
  float result = 0;
  int stride = input_tensor.numel() / dumpnum;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < input_tensor.numel(); i += stride) {
    result = paddle_mobile::fpga::fp16_2_fp32(data_tmp[i]);
    out << result << std::endl;
  }
  out.close();
  free(data_tmp);
}

void dump_stride_float(std::string filename, Tensor input_tensor,
                       const int dumpnum) {
  auto data_ptr = reinterpret_cast<float *>(input_tensor.get_data());
  std::ofstream out(filename.c_str());
  float result = 0;
  int stride = input_tensor.numel() / dumpnum;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < input_tensor.numel(); i += stride) {
    result = data_ptr[i];
    out << result << std::endl;
  }
  out.close();
}
static const char *g_resnet50 = "../models/resnet50";
const std::string g_image_src_float = "../images/image_src_float";  // NOLINT
int main() {
  paddle_mobile::fpga::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  if (paddle_mobile.Load(std::string(g_resnet50), true)) {
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 224, 224}, static_cast<float>(2),
                       static_cast<float>(2));
    readStream(g_image_src_float,
               input_tensor.mutable_data<float>({1, 3, 224, 224}));
    paddle_mobile.FeedData(input_tensor);
    paddle_mobile.Predict_To(-1);
    for (int i = 0; i < 73; i++) {
      auto tensor_ptr = paddle_mobile.FetchResult(i);
      std::string saveName = "resnet50_result_" + std::to_string(i);
      paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).get_data(),
                                           tensor_ptr->numel() * sizeof(half));
      // dump_stride_half(saveName, (*tensor_ptr), 20);
      // dump(saveName, (*tensor_ptr));
    }

    auto tensor_ptr = paddle_mobile.FetchResult(73);
    // dump_stride_float("resnet50_result_73", (*tensor_ptr), 20);
    tensor_ptr = paddle_mobile.FetchResult(74);
    // dump_stride_float("resnet50_result_74", (*tensor_ptr), 9999);

    float max = 0;
    auto data_ptr = tensor_ptr->data<float>();
    int maximumIdx = 0;
    for (int i = 0; i < (*tensor_ptr).numel(); i++) {
      if (data_ptr[i] > max) {
        maximumIdx = i;
        max = data_ptr[i];
      }
    }
    std::cout << "index : " << std::dec << maximumIdx << ",    value : " << max
              << std::endl;
    std::cout << "Computation done" << std::endl;
    return 0;
  }
}
