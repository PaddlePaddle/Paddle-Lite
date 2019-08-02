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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

bool read_file(std::vector<float>* result, const char* file_name) {
  std::ifstream infile(file_name);

  if (!infile.good()) {
    std::cout << "Cannot open " << file_name << std::endl;
    return false;
  }

  LOG(INFO) << "found filename: " << file_name;
  std::string line;

  while (std::getline(infile, line)) {
    (*result).push_back(static_cast<float>(atof(line.c_str())));
  }

  return true;
}

const char* bboxes_file = "multiclass_nms_bboxes_file.txt";
const char* scores_file = "multiclass_nms_scores_file.txt";
const char* out_file = "multiclass_nms_out_file.txt";

namespace paddle {
namespace lite {
class MulticlassNmsComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string bbox_ = "BBoxes";
  std::string conf_ = "Scores";
  std::string out_ = "Out";
  std::vector<int> priors_;
  int class_num_;
  int background_id_;
  int keep_topk_;
  int nms_topk_;
  float conf_thresh_;
  float nms_thresh_;
  float nms_eta_;
  bool share_location_;
  DDim bbox_dims_;
  DDim conf_dims_;

 public:
  MulticlassNmsComputeTester(const Place& place,
                             const std::string& alias,
                             std::vector<int> priors,
                             int class_num,
                             int background_id,
                             int keep_topk,
                             int nms_topk,
                             float conf_thresh,
                             float nms_thresh,
                             float nms_eta,
                             bool share_location,
                             DDim bbox_dims,
                             DDim conf_dims)
      : TestCase(place, alias),
        priors_(priors),
        class_num_(class_num),
        background_id_(background_id),
        keep_topk_(keep_topk),
        nms_topk_(nms_topk),
        conf_thresh_(conf_thresh),
        nms_thresh_(nms_thresh),
        nms_eta_(nms_eta),
        share_location_(share_location),
        bbox_dims_(bbox_dims),
        conf_dims_(conf_dims) {}

  void RunBaseline(Scope* scope) override {
    std::vector<float> vbbox;
    std::vector<float> vscore;
    std::vector<float> vout;

    if (!read_file(&vout, out_file)) {
      LOG(ERROR) << "load ground truth failed";
      return;
    }

    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(DDim({static_cast<int64_t>(vout.size() / 6), 6}));
    auto* out_data = out->mutable_data<float>();
    memcpy(out_data, vout.data(), vout.size() * sizeof(float));
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("multiclass_nms");
    op_desc->SetInput("BBoxes", {bbox_});
    op_desc->SetInput("Scores", {conf_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("priors", priors_);
    op_desc->SetAttr("class_num", class_num_);
    op_desc->SetAttr("background_id", background_id_);
    op_desc->SetAttr("keep_topk", keep_topk_);
    op_desc->SetAttr("nms_topk", nms_topk_);
    op_desc->SetAttr("conf_thresh", conf_thresh_);
    op_desc->SetAttr("nms_thresh", nms_thresh_);
    op_desc->SetAttr("nms_eta", nms_eta_);
    op_desc->SetAttr("share_location", share_location_);
  }

  void PrepareData() override {
    std::vector<float> bbox_data;
    std::vector<float> conf_data;

    if (!read_file(&bbox_data, bboxes_file)) {
      LOG(ERROR) << "load bbox file failed";
      return;
    }
    if (!read_file(&conf_data, scores_file)) {
      LOG(ERROR) << "load score file failed";
      return;
    }

    SetCommonTensor(bbox_, bbox_dims_, bbox_data.data());
    SetCommonTensor(conf_, conf_dims_, conf_data.data());
  }
};

TEST(MulticlassNms, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  int keep_top_k = 200;
  int nms_top_k = 400;
  float nms_eta = 1.;
  float score_threshold = 0.009999999776482582;
  int background_label = 0;
  float nms_threshold = 0.44999998807907104;
  int N = 1;
  int M = 1917;
  int class_num = 21;
  bool share_location = true;
  std::vector<int> priors(N, M);

  std::unique_ptr<arena::TestCase> tester(
      new MulticlassNmsComputeTester(place,
                                     "def",
                                     priors,
                                     class_num,
                                     background_label,
                                     keep_top_k,
                                     nms_top_k,
                                     score_threshold,
                                     nms_threshold,
                                     nms_eta,
                                     share_location,
                                     DDim({N, M, 4}),
                                     DDim({class_num, M, 4})));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle
