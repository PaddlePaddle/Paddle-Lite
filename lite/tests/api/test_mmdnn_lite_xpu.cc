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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_bool(perf, false, "perf?");
DEFINE_string(perf_input, "perf_input", "perf_input");
DEFINE_int32(perf_batch_size, 40, "perf_batch_size");
DEFINE_bool(use_xpu, true, "use_xpu?");
DEFINE_int32(perf_dev, 0, "perf_dev");

namespace paddle {
namespace lite {

class SampleReader {
 public:
  std::vector<std::vector<int64_t>> data;
  std::vector<std::vector<uint64_t>> lod;

  void Read() {
    std::string raw_input =
        "0 1;125 584 142 2114 197;125 756226 756913 855693 760836;125 584 142 "
        "2114 197 10 2899;125 756226 756913 855693 760836 10 750793;125 584 "
        "142 2114 197 10 2899 2 825 32 18499 125 584 295 2114 197 2114 2730 6 "
        "15 32 18499 125 584 142 295 2114 1423 21 2 334 863 5122 197 974 21 "
        "295 619 25 2114 1755 2701 197 15 216 23 18499 125 584 142 599 3228 23 "
        "2 5122 1917 804 5 2114 197 1236 3 2114 1403 15 3886 1080 23 1150 125 "
        "475 23 2998 23;125 756226 756913 855693 760836 10 750793 2 825 750355 "
        "18499 881680 756226 295 765124 760836 2114 872813 754265 15 32 18499 "
        "881680 756226 756913 761251 765124 752843 766823 2 334 759834 5122 "
        "774643 758458 21 295 755114 25 1148365 1755 2701 197 15 216 23 18499 "
        "881680 756226 756913 826848 3228 23 2 5122 831009 804 752371 2114 "
        "760836 1236 3 2114 910393 15 3886 1080 23 877375 752137 761034 792123 "
        "2998 23;1;1;\n"
        "0 0;125 584 142 2114 197;125 756226 756913 855693 760836;121 28 1054 "
        "1459 125 72 32 2321 531 125 295 584 142 2114 197 14 477 30 121;121 28 "
        "764114 1459 753052 750694 750001 886192 750435 752179 295 584 756913 "
        "855693 760836 14 477 30 753504;121 28 1054 1459 125 72 32 2321 531 "
        "125 295 584 142 2114 197 2 121 28 1054 1459 125 72 32 2321 531 125 "
        "295 584 142 4 263 2114 197 43 95 863 2114 323 20 142 626 11 2 45 10 "
        "45 58 142 65 918 741 2114 197 764 3 5122 26 51 1266 2037 295 222 1121 "
        "4491 3 545 4338 11 2 5122 26 495 3 142 3444 3249 2114 197 3 626 4 "
        "2794;121 28 764114 1459 753052 750694 750001 886192 750435 752179 295 "
        "584 756913 855693 760836 2 121 28 764114 1459 753052 750694 750001 "
        "886192 750435 752179 295 584 756913 4 750885 2114 760836 43 750030 "
        "754302 2114 323 822131 142 626 769001 2 45 750128 750324 58 142 "
        "1147454 918 910829 2114 760836 841946 767340 5122 779102 51 1266 2037 "
        "756461 222 752031 942669 1139389 780275 4338 830597 2 5122 779102 495 "
        "761418 142 3444 852932 2114 760836 3 760162 757966 751127;121 295 "
        "5593 142 2114 197;121 295 5593 925208 2114 760836;\n"
        "0 0;125 584 142 2114 197;125 756226 756913 855693 760836;207 125 584 "
        "142 2114 1423 14 5283 1745 73;207 752276 756226 756913 855693 752843 "
        "14 5283 781651 786597;6109 18807 142 5 64 5283 1745 73 3690 1060 3626 "
        "4 716 51 1030 2114 197 4 428 936 9066 10 10 10 2 207 125 584 142 2114 "
        "1423 2 15329 2114 197 5669 401 318 285 953 4 2114 197 2285 7 1783 11 "
        "2 5122 197 14017 584;6109 18807 142 5 755319 5283 781651 786597 3690 "
        "1060 3626 4 716 910478 1030 2114 760836 4 750323 936 9066 10 750002 "
        "750002 2 207 752276 756226 756913 855693 752843 2 15329 2114 760836 "
        "5669 401 318 757541 750261 4 2114 760836 2285 7 757639 11 2 5122 "
        "774643 14017 584;125 584 142 1745 5122;125 756226 756913 1745 "
        "755836;\n"
        "0 0;125 584 142 2114 197;125 756226 756913 855693 760836;149 396 778 "
        "584 142 295 2114 1423 14 64 125 584 73 21 36670 5834 10 211 25;149 "
        "751876 1048872 584 756913 761251 765124 752843 14 64 125 756226 73 "
        "944567 36670 5834 10 750012 753240;101 10 2114 197 3 946 2 149 396 "
        "778 584 142 295 2114 1423 2 2610 6 1444 111 2114 948 72 32 21 15 494 "
        "25 4 2114 197 5669 1145 2 148 295 149 396 778 584 142 295 21 22853 41 "
        "348 619 25 366 5305 2114 807 4 1115 381 1955 2114 11;101 751178 2114 "
        "760836 3 946 2 149 751876 1048872 584 756913 761251 765124 752843 2 "
        "2610 753567 775165 750899 972788 948 750125 750001 751875 15 494 25 4 "
        "2114 760836 5669 1145 2 148 808886 982157 751876 1048872 584 756913 "
        "761251 790772 22853 41 348 619 25 366 894206 2114 1008440 4 753953 "
        "381 851474 765868 11;149 396 778 584 142 295 2 149 396 354 778 584 "
        "142 1333 2 584 778 295 5122 2 149 396 778 584 3609 2 149 396 64478 "
        "816 14246 1423 2 149 396 584 32 127 19 3609 2 149 396 584 73 2 149 "
        "396 584 778 295 2285 142 4922 323 2 149 396 584 2114 2 149 396 253 "
        "584 2114 197;149 751876 1048872 584 756913 761251 2 149 751876 756286 "
        "767182 584 756913 1333 2 584 778 897778 941364 2 149 751876 1048872 "
        "584 1102835 2 149 751876 64478 816 14246 912094 2 149 751876 584 "
        "773547 127 750771 791456 2 149 751876 584 73 2 149 751876 584 778 "
        "897778 2285 751493 791984 323 2 149 751876 584 2114 2 149 751876 "
        "808443 835481 2114 760836;\n"
        "0 0;125 584 142 2114 197;125 756226 756913 855693 760836;125 584 545 "
        "149 14 125 584;125 756226 545 874302 14 125 756226;2204 25 30 1692 "
        "1770 6534 295 125 584 72 32 1346 4 2698 2114 197 11 2 4235 4301 240 "
        "295 125 584 72 32 21 6708 15 56974 494 25 1030 2114 197 110 804 495 "
        "611 2 221 759 341 6 5283 1745 73 71 2114 1423 71 125 584 545 149 149 "
        "2 505 345 58 125 584 65 3486 2114 295 4 45 786 196 6604 6086;2204 25 "
        "30 797189 1770 1191824 295 752782 756226 751697 750001 1346 4 2698 "
        "2114 760836 765158 2 4235 4301 240 753859 752782 756226 751697 750001 "
        "751875 6708 15 56974 494 25 1030 2114 760836 777607 762850 966521 611 "
        "2 221 752565 750130 750084 910219 781651 786597 71 2114 752843 71 125 "
        "756226 545 874302 149 2 505 825657 782848 125 756226 65 3486 2114 "
        "760669 4 45 755747 758903 6604 6086;125 584 2114 2 125 584 2114 1423 "
        "2 125 584 2114 149 2 149 584 1745 5122 725 2 2114 125 584 2 125 584 "
        "2114 2 2621 584 2114 2 527 37 2754 130 170 1013 494 887 240 2 4521 "
        "11111 586 2321 531 125 584 142 1360 816 2842 1423 2 125 584 2114;125 "
        "756226 2114 2 125 756226 2114 752843 2 125 756226 2114 783644 2 149 "
        "760183 1745 755836 725 2 2114 125 756226 2 125 756226 2114 2 2621 "
        "932600 2114 2 527 751304 869964 754462 170 1013 750719 778287 774620 "
        "2 4521 11111 586 2321 750435 752179 756226 756913 1360 764399 2842 "
        "1423 2 125 756226 2114;\n"
        "0 0;125 584 142 2114 197;125 756226 756913 855693 760836;207 584 142 "
        "2114 197 4 207 584 142 2114 197 674 14 240 4328 14 4328 767;207 "
        "1237071 756913 855693 760836 4 207 1237071 756913 855693 760836 674 "
        "14 240 755573 14 4328 795065;207 584 142 2114 197 2 325 71 71 207 584 "
        "142 2114 197 2 876 125 140 2114 197 2 207 584 142 2114 197 674 1210 "
        "239 4328 767 268 1349 485 28 4389 504 3 941 57 1419 1978 11;207 "
        "1237071 756913 855693 760836 2 325 71 71 207 1237071 756913 855693 "
        "760836 2 876 125 750977 1250790 760836 2 207 1237071 756913 855693 "
        "760836 674 814792 755820 812174 795065 818859 817155 816597 761001 "
        "774461 780904 820475 1109800 790141 790459 780324 770390;584 142 295 "
        "2114 232 2 207 584 2114 197 2 584 142 295 2114 232 2 584 142 512 2114 "
        "197;584 756913 761251 765124 1006359 2 207 1237071 2114 760836 2 584 "
        "756913 761251 765124 1006359 2 584 756913 879930 2114 760836;";

    auto lines = Split(raw_input, "\n");
    for (auto& line : lines) {
      auto split1 = Split(line, ";");
      if (data.size() == 0) {
        for (size_t i = 1; i < split1.size(); ++i) {
          data.push_back(std::vector<int64_t>());
          lod.push_back({0});
        }
      }

      for (size_t i = 1; i < split1.size(); ++i) {
        auto split2 = Split(split1[i], " ");
        if (split2.size() == 0) {
          split2.push_back("1280000");
        }
        for (auto e : split2) {
          data[i - 1].push_back(std::stoi(e.c_str(), nullptr, 0));
        }
        lod[i - 1].push_back(lod[i - 1].back() + split2.size());
      }
    }
  }
};

class FileReader {
  std::ifstream ifs;

 public:
  std::vector<std::vector<int64_t>> data;
  std::vector<std::vector<uint64_t>> lod;

  void Init(std::string file_name) { ifs.open(file_name); }

  int Read(int maxline) {
    data.clear();
    lod.clear();

    std::string line;
    int cnt = 0;
    while (cnt < maxline && getline(ifs, line)) {
      std::vector<std::string> split1 = Split(line, ";");
      if (data.size() == 0) {
        for (size_t i = 1; i < split1.size(); ++i) {
          data.push_back(std::vector<int64_t>());
          lod.push_back({0});
        }
      }

      for (size_t i = 1; i < split1.size(); i++) {
        std::vector<std::string> split2 = Split(split1[i], " ");
        if (split2.size() == 0) {
          split2.push_back("1280000");
        }
        for (size_t j = 0; j < split2.size(); j++) {
          data[i - 1].push_back(std::stoi(split2[j].c_str(), nullptr, 0));
        }
        lod[i - 1].push_back(lod[i - 1].back() + split2.size());
      }
      cnt++;
    }
    return cnt;
  }
};

TEST(MMDNN, test_mmdnn_lite_xpu) {
  lite_api::CxxConfig config;
  // config.set_model_dir(FLAGS_model_dir);
  config.set_model_file(FLAGS_model_dir + "/__model__");
  config.set_param_file(FLAGS_model_dir + "/__param__");
  config.set_xpu_dev_per_thread(FLAGS_perf_dev);
  if (FLAGS_use_xpu) {
    config.set_valid_places(
        {lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
         lite_api::Place{TARGET(kXPU), PRECISION(kInt64)},
         lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
         lite_api::Place{TARGET(kX86), PRECISION(kInt64)},
         lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  } else {
    config.set_valid_places(
        {lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
         lite_api::Place{TARGET(kX86), PRECISION(kInt64)},
         lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  }
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  if (FLAGS_perf) {
    FileReader file_reader;
    file_reader.Init(FLAGS_perf_input);
    int UB_batch = FLAGS_perf_batch_size;  //  upper bound of batch
    int iter = 0;
    double tsc_sum = 0;

    while (true) {
      int batch = file_reader.Read(UB_batch);
      if (batch <= 0) {
        break;
      }
      ++iter;
      for (size_t i = 0; i < file_reader.data.size(); ++i) {
        auto input_x = predictor->GetInput(i);
        input_x->Resize({(int64_t)file_reader.data[i].size(), 1});
        input_x->SetLoD({file_reader.lod[i]});
        auto* data_x = input_x->mutable_data<int64_t>();
        memcpy(data_x,
               file_reader.data[i].data(),
               file_reader.data[i].size() * sizeof(int64_t));
      }

      auto start = GetCurrentUS();
      predictor->Run();
      auto end = GetCurrentUS();
      tsc_sum += end - start;
    }
    LOG(INFO) << "================== Speed Report ===================";
    LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num "
              << FLAGS_threads << ", warmup: " << FLAGS_warmup
              << ", repeats: " << iter << ", spend " << tsc_sum / iter / 1000.0
              << " ms in average.";

    return;
  }

  SampleReader sample_reader;
  sample_reader.Read();

  for (size_t i = 0; i < sample_reader.data.size(); ++i) {
    auto input_x = predictor->GetInput(i);
    input_x->Resize({(int64_t)sample_reader.data[i].size(), 1});
    input_x->SetLoD({sample_reader.lod[i]});
    auto* data_x = input_x->mutable_data<int64_t>();
    memcpy(data_x,
           sample_reader.data[i].data(),
           sample_reader.data[i].size() * sizeof(int64_t));
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  auto out = predictor->GetOutput(0);
  auto out_shape = out->shape();
  auto out_size = std::accumulate(
      out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
  for (int i = 0; i < out_size; ++i) {
    LOG(INFO) << "out[" << i << "] = " << out->data<float>()[i];
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
