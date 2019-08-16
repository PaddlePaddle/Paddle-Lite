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

    std::vector<int64_t> ids{1918, 117, 55, 97, 1352, 4272, 1656, 903};

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
      paddle_mobile.Predict(words);
      DLOG << *paddle_mobile.Fetch();
    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 1 << "ms"
              << std::endl;
  }

  auto time2 = time();
  std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;
  //    1064 1603 644 699 2878 1219 867 1352 8 1 13 312 479

  std::vector<int64_t> ids{
      2084, 635,  1035, 197,  990,  150,  1132, 2403, 546,  770,  4060, 3352,
      1798, 1589, 1352, 98,   136,  3461, 3186, 1159, 515,  764,  278,  1178,
      5044, 4060, 943,  932,  463,  1198, 3352, 374,  1198, 3352, 374,  2047,
      1069, 1589, 3672, 1178, 1178, 2165, 1178, 2084, 635,  3087, 2236, 546,
      2047, 1549, 546,  2047, 302,  2202, 398,  804,  397,  657,  804,  866,
      932,  2084, 515,  2165, 397,  302,  2202, 526,  992,  906,  1215, 1589,
      4493, 2403, 723,  932,  2084, 635,  1352, 932,  444,  2047, 1159, 1893,
      1579, 59,   330,  98,   1296, 1159, 3430, 738,  3186, 1071, 2174, 3933};

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
    paddle_mobile.Predict(words);
    DLOG << *paddle_mobile.Fetch();
  }
  auto time4 = time();
  std::cout << "predict cost :" << time_diff(time3, time4) / 1 << "ms"
            << std::endl;
  return 0;
}
