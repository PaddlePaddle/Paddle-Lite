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

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << std::endl
              << "./test_benchmark fluid_model feed_shape thread_num [use_fuse]"
              << std::endl;
    std::cout << "use_fuse: optional, bool, default is 1\n";
    return 1;
  }
  bool optimize = true;
  char* fluid_model = argv[1];
  char* feed_shape = argv[2];
  int thread_num = atoi(argv[3]);
  if (argc == 5) {
    optimize = atoi(argv[4]);
  }

  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(thread_num);
  auto time1 = time();
  //  if (paddle_mobile.Load(fluid_model, optimize, false, 1, true)) {
  if (paddle_mobile.Load(std::string(fluid_model) + "/model",
                         std::string(fluid_model) + "/params", optimize, false,
                         1, true)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time2) << "ms\n";
    paddle_mobile::framework::Tensor input;
    std::shared_ptr<paddle_mobile::framework::Tensor> output;
    std::vector<int64_t> dims{1, 3, 224, 224};
    if (feed_shape) {
      sscanf(feed_shape, "%ld,%ld,%ld,%ld", &dims[0], &dims[1], &dims[2],
             &dims[3]);
    }
    std::cout << "feed shape: [" << dims[0] << ", " << dims[1] << ", "
              << dims[2] << ", " << dims[3] << "]\n";
    paddle_mobile::framework::DDim in_shape =
        paddle_mobile::framework::make_ddim(dims);
    SetupTensor<float>(&input, in_shape, 0.f, 255.f);
    // warmup
    for (int i = 0; i < 2; ++i) {
      paddle_mobile.Predict(input);
    }
    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      paddle_mobile.Predict(input);
    }

    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 10 << "ms\n";
    std::ostringstream os("output tensor size: ");
    output = paddle_mobile.Fetch();
    os << output->numel() << "\n" << output->data<float>()[0];
    for (int i = 1; i < output->numel(); ++i) {
      os << ", " << output->data<float>()[i];
    }
    std::string output_str = os.str();
    //    std::cout << output_str << std::endl;
  }
  return 0;
}
