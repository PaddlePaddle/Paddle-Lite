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

void readStream(std::string filename, uint8_t *buf) {
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

static const char *g_rfcn_combine = "../models/rfcn";
static const char *g_image_src_float = "../models/rfcn/data.bin";
int main() {
  paddle_mobile::fpga::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;

  if (paddle_mobile.Load(std::string(g_rfcn_combine) + "/model",
                         std::string(g_rfcn_combine) + "/params", true, false,
                         1, true)) {
    float img_info[3] = {768, 1536, 768.0f / 960.0f};
    auto img = fpga::fpga_malloc(768 * 1536 * 3 * sizeof(float));
    readStream(g_image_src_float, reinterpret_cast<uint8_t *>(img));
    std::vector<void *> v(3, nullptr);
    paddle_mobile.FeedData({img_info, img});
    paddle_mobile.Predict_To(-1);
    paddle_mobile.GetResults(&v);
    DLOG << "Computation done";
    fpga::fpga_free(img);
  }

  return 0;
}
