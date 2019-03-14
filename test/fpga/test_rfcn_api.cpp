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

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"

#ifdef PADDLE_MOBILE_FPGA_V1
#include "fpga/V1/api.h"
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#endif

#include <string>

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
  DLOG << length;
  in.close();
}

void convert_to_chw(int16_t **data_in, int channel, int height, int width,
                    int num, int16_t *data_tmp) {
  int64_t amount_per_side = width * height;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          *(data_tmp + n * amount_per_side * channel + c * amount_per_side +
            width * h + w) = *((*data_in)++);
        }
      }
    }
  }
}

void dump_stride_half(std::string filename, Tensor input_tensor,
                      const int dumpnum, bool use_chw) {
  // bool use_chw = true;
  if (input_tensor.dims().size() != 4) return;
  int c = (input_tensor.dims())[1];
  int h = (input_tensor.dims())[2];
  int w = (input_tensor.dims())[3];
  int n = (input_tensor.dims())[0];
  auto data_ptr = input_tensor.get_data();
  auto *data_ptr_16 = reinterpret_cast<half *>(data_ptr);
  auto data_tmp = data_ptr_16;
  if (use_chw) {
    data_tmp =
        reinterpret_cast<half *>(malloc(n * c * h * w * sizeof(int16_t)));
    convert_to_chw(&data_ptr_16, c, h, w, n, data_tmp);
  }
  std::ofstream out(filename.c_str());
  float result = 0;
  int stride = input_tensor.numel() / dumpnum;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < input_tensor.numel(); i += stride) {
    result = paddle_mobile::fpga::fp16_2_fp32(data_tmp[i]);
    out << result << std::endl;
  }
  out.close();
  if (data_tmp != data_ptr_16) {
    free(data_tmp);
  }
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

void dump_stride(std::string filename, Tensor input_tensor, const int dumpnum,
                 bool use_chw) {
  static int i = 0;
  if (input_tensor.numel() == 0) {
    return;
  }
  if (input_tensor.type() == typeid(float)) {
    DLOG << "op: " << i++ << ", float data  " << input_tensor.numel();

    dump_stride_float(filename, input_tensor, dumpnum);
  } else {
    DLOG << "op: " << i++ << ", half data  " << input_tensor.numel();

    dump_stride_half(filename, input_tensor, dumpnum, use_chw);
  }
  DLOG << "dump input address: " << input_tensor.get_data();
}

static const char *g_rfcn_combine = "../models/rfcn";
static const char *g_image_src_float = "../models/rfcn/data.bin";
int main() {
  paddle_mobile::fpga::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;

  if (paddle_mobile.Load(std::string(g_rfcn_combine) + "/model",
                         std::string(g_rfcn_combine) + "/params", true, false,
                         1, true)) {
    float img_info[3] = {768, 1536, 768.0f / 960.0f};
    auto img = reinterpret_cast<float *>(
        fpga::fpga_malloc(768 * 1536 * 3 * sizeof(float)));
    readStream(g_image_src_float, reinterpret_cast<char *>(img));

    std::vector<void *> v(3, nullptr);
    paddle_mobile.FeedData(std::vector<void *>({img_info, img}));
    paddle_mobile.Predict_To(-1);

    for (int i = 65; i < 69; i++) {
      auto tensor_ptr = paddle_mobile.FetchResult(i);
      std::string saveName = "rfcn_" + std::to_string(i);
      paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).get_data(),
                                           tensor_ptr->numel() * sizeof(float));
      dump_stride(saveName, (*tensor_ptr), tensor_ptr->numel(), true);
    }
    //   paddle_mobile.GetResults(&v);
    DLOG << "Computation done";
    fpga::fpga_free(img);
  }

  return 0;
}
