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

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  auto time1 = time();
  //  auto isok = paddle_mobile.Load(std::string(g_mobilenet_detect) + "/model",
  //                     std::string(g_mobilenet_detect) + "/params", true);

  auto isok = paddle_mobile.Load(g_nlp, true, false, 1, true);

  //  auto isok = paddle_mobile.Load(std::string(g_nlp) + "/model",
  //                                 std::string(g_nlp) + "/params", false);
  if (isok) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;
    //    1064 1603 644 699 2878 1219 867 1352 8 1 13 312 479

    std::vector<int64_t> ids{1064, 1603, 644, 699, 2878, 1219, 867,
                             1352, 8,    1,   13,  312,  479};

    paddle_mobile::framework::LoDTensor words;
    auto size = static_cast<int>(ids.size());
    paddle_mobile::framework::LoD lod{{0, ids.size()}};
    DDim dims{size, 1};
    words.Resize(dims);
    words.set_lod(lod);
    DLOG << "words lod : " << words.lod();
    auto *pdata = words.mutable_data<int64_t>();
    size_t n = words.numel() * sizeof(int64_t);
    DLOG << "n :" << n;
    memcpy(pdata, ids.data(), n);
    DLOG << "words lod 22: " << words.lod();
    auto time3 = time();
    for (int i = 0; i < 1; ++i) {
      auto vec_result = paddle_mobile.PredictLod(words);
      DLOG << *vec_result;
    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 1 << "ms"
              << std::endl;
  }
  return 0;
}
