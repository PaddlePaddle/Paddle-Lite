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
#include <sstream>
#include "../test_helper.h"
#include "../test_include.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: ./test_googlenet fluid-model input-image image-shape "
                 "[thread-num] [fusion]\n"
              << " fluid-model: fluid model path. \n"
              << " input-image: input raw image path. \n"
              << " image-shape: input tensor shape, such as 1,3,224,224.\n"
              << " thread-num: optional int, threads count, default is 1.\n"
              << " fusion: optional bool, default is 0.\n";
    return 1;
  }
  int thread_num = 1;
  bool optimize = false;
  char *fluid_model = argv[1];
  char *input_img = argv[2];
  char *feed_shape = argv[3];
  if (argc >= 5) {
    thread_num = atoi(argv[4]);
  }
  if (argc >= 6) {
    optimize = atoi(argv[5]);
  }
#ifdef PADDLE_MOBILE_FPGA
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
#endif
#ifdef PADDLE_MOBILE_CPU
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
#endif
  paddle_mobile.SetThreadNum(thread_num);
  auto time1 = time();
  std::vector<float> output;
  if (paddle_mobile.Load(fluid_model, optimize, false, 1, true)) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;
    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 224, 224};
    if (feed_shape) {
      sscanf(feed_shape, "%lld,%lld,%lld,%lld", &dims[0], &dims[1], &dims[2],
             &dims[3]);
    }
    std::cout << "feed shape: [" << dims[0] << ", " << dims[1] << ", "
              << dims[2] << ", " << dims[3] << "]" << std::endl;

    GetInput<float>(input_img, &input, dims);

    // warmup
    for (int i = 0; i < 10; ++i) {
      output = paddle_mobile.Predict(input, dims);
    }
    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      output = paddle_mobile.Predict(input, dims);
    }
    auto time4 = time();
    std::cout << "predict cost: " << time_diff(time3, time4) / 10 << "ms\n";

    std::ostringstream os;
    os << output[0];
    for (int i = 1; i < output.size(); ++i) {
      os << ", " << output[i];
    }
    DLOG << os.str();
  }
  return 0;
}
