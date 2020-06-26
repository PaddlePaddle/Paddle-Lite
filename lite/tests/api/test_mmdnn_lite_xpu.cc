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

namespace paddle {
namespace lite {

std::vector<int64_t> input0;
std::vector<uint64_t> input0_lod = {0};
std::vector<int64_t> input1;
std::vector<uint64_t> input1_lod = {0};
std::vector<int64_t> input2;
std::vector<uint64_t> input2_lod = {0};
std::vector<int64_t> input3;
std::vector<uint64_t> input3_lod = {0};
std::vector<int64_t> input4;
std::vector<uint64_t> input4_lod = {0};
std::vector<int64_t> input5;
std::vector<uint64_t> input5_lod = {0};

void parse_input() {
  std::string raw_input =
      "0 1;145 10251 839 3719 428 52;1050488 1050488 911898 3719 760166 "
      "760166;3719 428 52 18 1102 10327 252 20 153 2897 1146 70 156 6 145 "
      "10251 839 5 1779 1729 1779 1729 18 2707 6 2707 20 4742 4937 432 6 "
      "3869;3719 760166 760166 18 1035176 1035176 764393 764393 1259006 767614 "
      "767614 1020808 769579 793958 793958 1050488 911898 751332 751332 750336 "
      "750799 750336 751575 751575 751544 751735 751397 751365 751512 751512 "
      "753011 751562;3719 428 52 18 1102 10327 252 20 153 2897 1146 70 156 6 "
      "145 10251 839 2 1211 3 3719 720 1540 145 10251 839 9405 4315 5998 4 2 "
      "600 373 41 3719 428 52 44 10251 4302 1319 7 12 2 768 6 918 6 841 870 8 "
      "843 8 271;3719 760166 760166 18 1035176 1035176 764393 764393 1259006 "
      "767614 767614 1020808 769579 793958 793958 1050488 911898 2 773899 "
      "773899 3719 1118420 1118420 1050488 1050488 911898 9405 4315 5998 4 2 "
      "785435 785435 41 3719 760166 760166 44 10251 4302 1319 750118 750118 2 "
      "750465 750465 750274 750398 750233 751252 751252 753447 752830 753112;\n"
      "0 0;145 10251 839 3719 428 52;1050488 1050488 911898 3719 760166 "
      "760166;2109 2467 1805 227 3719 428 52 18 1102 10327 252 20 6 242 78 6 "
      "532 78;2109 2467 1805 1245431 1245431 760166 760166 18 1035176 1035176 "
      "764393 764393 752116 242 750370 750370 752081 751247;2109 2467 1805 227 "
      "3719 428 52 18 1102 10327 252 20 2 145 242 1050 252 3582 2212;2109 2467 "
      "1805 1245431 1245431 760166 760166 18 1035176 1035176 764393 764393 2 "
      "871717 871717 757921 757921 3582 2212;\n"
      "0 0;145 10251 839 3719 428 52;1050488 1050488 911898 3719 760166 "
      "760166;145 10251 839 76 31 1337 823 7506 567 65 170 8 21293 3719 5 43 "
      "394 743 42;1050488 1050488 911898 750016 750016 1337 823 7506 762617 "
      "762617 866652 8 21293 3719 5 43 914758 914758 757202;145 10251 839 76 "
      "31 1337 823 7506 567 65 170 8 21293 3719 2 17580 30 523324 3 10251 4104 "
      "281 3 8511 3719 2217 3 13 226 3083 4 11251 1606 357 9 2 145 10251 839 "
      "76 31 1337 823 7506 567 65 170 2 7506 2445 8 145 10251 839 528 839 "
      "19670 6538;1050488 1050488 911898 750016 750016 1337 823 7506 762617 "
      "762617 866652 8 21293 3719 2 816626 816626 523324 3 1181698 1181698 "
      "751656 780821 1063148 3719 2217 3 752498 752498 831323 753602 11251 "
      "1606 357 9 2 1050488 1050488 911898 750016 750016 1337 823 7506 762617 "
      "762617 866652 2 7506 753045 753045 756756 1050488 911898 528 839 19670 "
      "6538;\n"
      "0 0;145 10251 839 3719 428 52;1050488 1050488 911898 3719 760166 "
      "760166;145 10251 839 99 4 1102 10327 2196 41 3719 428 52 44 99 4 2899 "
      "229 10 10 10;1050488 1050488 911898 807966 750273 1035176 1035176 "
      "1237875 41 3719 760166 760166 753645 753645 750273 2899 229 750001 "
      "750001 750001;145 10251 839 99 4 1102 10327 2196 41 3719 428 52 44 99 4 "
      "2899 229 10 10 10 2 1177 8 145 10251 839 99 4 1102 10327 2196 41 3719 "
      "428 52 44 99 4 2 101 8 1922 17 2184 2 1154 1922 72 1198 1266 "
      "4516;1050488 1050488 911898 807966 750273 1035176 1035176 1237875 41 "
      "3719 760166 760166 753645 753645 750273 2899 229 750001 750001 750001 2 "
      "750257 750257 756756 1050488 911898 807966 750273 1035176 1035176 "
      "1237875 41 3719 760166 760166 753645 753645 750273 2 764513 764513 "
      "851213 851213 854628 2 753018 753018 754317 753328 754085 754070;\n"
      "0 0;145 10251 839 3719 428 52;1050488 1050488 911898 3719 760166 "
      "760166;73 5347 112 8 145 10251 839 262 169 22729 3719 6 743 6 339 1156 "
      "78 136 399 693 128 571;776150 776150 112 756756 756756 1050488 911898 "
      "791355 791355 22729 3719 6 758277 758277 750137 750234 750241 750178 "
      "750055 750216 750212 750049;73 5347 112 8 145 10251 839 262 169 22729 "
      "3719 2 588 415 549 415 115 23;776150 776150 112 756756 756756 1050488 "
      "911898 791355 791355 22729 3719 2 750221 750221 750262 750277 750277 "
      "750261;";
  auto raw_lines = Split(raw_input, "\n");
  for (auto& raw_line : raw_lines) {
    auto inputx = Split(raw_line, ";");
    for (size_t i = 1; i < inputx.size(); ++i) {
      auto tokens = Split(inputx[i], " ");
      static std::vector<int64_t>* const input_array[] = {
          &input0, &input0, &input1, &input2, &input3, &input4, &input5};
      static std::vector<uint64_t>* const lod_array[] = {&input0_lod,
                                                         &input0_lod,
                                                         &input1_lod,
                                                         &input2_lod,
                                                         &input3_lod,
                                                         &input4_lod,
                                                         &input5_lod};
      for (auto token : tokens) {
        input_array[i]->push_back((int64_t)atoi(token.c_str()));
      }
      lod_array[i]->push_back((uint64_t)tokens.size() +
                              (*lod_array[i])[lod_array[i]->size() - 1]);
    }
  }
  return;
}

class mmdnn_reader {
  std::ifstream inF;
  std::vector<std::string> string_split(const std::string& in,
                                        const std::string& delim) {
    std::vector<std::string> ret;
    if (in == "") {
      return ret;
    }
    auto begpos = in.find_first_not_of(delim);
    while (begpos != std::string::npos) {
      auto endpos = in.find_first_of(delim, begpos);
      if (endpos == std::string::npos) {
        endpos = in.size();
      }
      std::string ssubstr = in.substr(begpos, endpos - begpos);
      ret.push_back(ssubstr);
      begpos = endpos + 1;
      if (endpos >= (in.size() - 1)) {
        break;
      }
    }
    return ret;
  }

 public:
  std::vector<int64_t> data[6];
  std::vector<uint64_t> lod[6];

  void init(std::string file_name) { inF.open(file_name); }

  int read(int maxline) {
    for (int i = 0; i < 6; i++) {
      data[i].clear();
    }
    for (int i = 0; i < 6; i++) {
      lod[i].clear();
      lod[i].push_back(0);
    }
    std::string line;
    int cnt = 0;
    while (cnt < maxline && getline(inF, line)) {
      std::vector<std::string> split1 = string_split(line, ";");
      for (int i = 1; i < 7; i++) {
        std::vector<std::string> split2 = string_split(split1[i], " ");
        if (split2.size() == 0) {
          split2.push_back("1280000");
        }
        for (size_t j = 0; j < split2.size(); j++) {
          data[i - 1].push_back(std::stoi(split2[j].c_str(), nullptr, 0));
        }
        // if (i % 2 == 1) {
        // lod[i / 2].push_back(lod[i / 2].back() + split2.size());
        //}
        lod[i - 1].push_back(lod[i - 1].back() + split2.size());
      }
      cnt++;
    }
    return cnt;
  }
};

TEST(MMDNN, test_mmdnn_lite_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kXPU), PRECISION(kInt64)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kInt64)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  if (FLAGS_perf) {
    mmdnn_reader reader;
    reader.init(FLAGS_perf_input);
    int UB_batch = 40;  //  upper bound of batch
    int iter = 0;
    double tsc_sum = 0;

    while (true) {
      int batch = reader.read(UB_batch);
      if (batch <= 0) {
        break;
      }
      ++iter;
      for (int i = 0; i < 6; ++i) {
        auto input_x = predictor->GetInput(i);
        input_x->Resize({(int64_t)reader.data[i].size(), 1});
        input_x->SetLoD({reader.lod[i]});
        auto* data_x = input_x->mutable_data<int64_t>();
        memcpy(data_x,
               reader.data[i].data(),
               reader.data[i].size() * sizeof(int64_t));
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

  parse_input();

  {
    std::vector<int64_t> input0_shape{(int64_t)input0.size(), 1};
    auto input_tensor0 = predictor->GetInput(0);
    input_tensor0->Resize(input0_shape);
    input_tensor0->SetLoD({input0_lod});
    auto* data0 = input_tensor0->mutable_data<int64_t>();
    memcpy(data0, input0.data(), sizeof(int64_t) * input0.size());
  }
  {
    std::vector<int64_t> input1_shape{(int64_t)input1.size(), 1};
    auto input_tensor1 = predictor->GetInput(1);
    input_tensor1->Resize(input1_shape);
    input_tensor1->SetLoD({input1_lod});
    auto* data1 = input_tensor1->mutable_data<int64_t>();
    memcpy(data1, input1.data(), sizeof(int64_t) * input1.size());
  }
  {
    std::vector<int64_t> input2_shape{(int64_t)input2.size(), 1};
    auto input_tensor2 = predictor->GetInput(2);
    input_tensor2->Resize(input2_shape);
    input_tensor2->SetLoD({input2_lod});
    auto* data2 = input_tensor2->mutable_data<int64_t>();
    memcpy(data2, input2.data(), sizeof(int64_t) * input2.size());
  }
  {
    std::vector<int64_t> input3_shape{(int64_t)input3.size(), 1};
    auto input_tensor3 = predictor->GetInput(3);
    input_tensor3->Resize(input3_shape);
    input_tensor3->SetLoD({input3_lod});
    auto* data3 = input_tensor3->mutable_data<int64_t>();
    memcpy(data3, input3.data(), sizeof(int64_t) * input3.size());
  }
  {
    std::vector<int64_t> input4_shape{(int64_t)input4.size(), 1};
    auto input_tensor4 = predictor->GetInput(4);
    input_tensor4->Resize(input4_shape);
    input_tensor4->SetLoD({input4_lod});
    auto* data4 = input_tensor4->mutable_data<int64_t>();
    memcpy(data4, input4.data(), sizeof(int64_t) * input4.size());
  }
  {
    std::vector<int64_t> input5_shape{(int64_t)input5.size(), 1};
    auto input_tensor5 = predictor->GetInput(5);
    input_tensor5->Resize(input5_shape);
    input_tensor5->SetLoD({input5_lod});
    auto* data5 = input_tensor5->mutable_data<int64_t>();
    memcpy(data5, input5.data(), sizeof(int64_t) * input5.size());
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
