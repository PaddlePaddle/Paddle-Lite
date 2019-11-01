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
#include <fstream>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

class GenerateProposalsComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string Scores_ = "Scores";
  std::string BboxDeltas_ = "BboxDeltas";
  std::string ImInfo_ = "ImInfo";
  std::string Anchors_ = "Anchors";
  std::string Variances_ = "Variances";
  int pre_nms_topN_ = 6000;
  int post_nms_topN_ = 1000;
  float nms_thresh_ = 0.699999988079071;
  float min_size_ = 0.0;
  float eta_ = 1.0;
  std::string RpnRois_ = "RpnRois";
  std::string RpnRoiProbs_ = "RpnRoiProbs";

 public:
  GenerateProposalsComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* rois = scope->NewTensor(RpnRois_);
    auto* probs = scope->NewTensor(RpnRoiProbs_);
    CHECK(rois);
    CHECK(probs);
    rois->Resize(std::vector<int64_t>({304, 4}));
    probs->Resize(std::vector<int64_t>({304, 1}));
    std::vector<uint64_t> lod0({0, 152, 304});
    LoD lod;
    lod.push_back(lod0);
    rois->set_lod(lod);
    probs->set_lod(lod);

    auto* rois_data = rois->mutable_data<float>();
    auto* probs_data = probs->mutable_data<float>();

    std::string base_path = "/data/local/tmp/data_files/";
    std::string filename;
    std::ifstream reader;
    // rois
    filename = "result_generate_proposals_0.tmp_0.txt";
    reader.open(base_path + filename);
    for (int i = 0; i < rois->numel(); i++) {
      reader >> rois_data[i];
    }
    LOG(INFO) << "Read Rois data." << rois_data[0];
    reader.close();
    // probs
    filename = "result_generate_proposals_0.tmp_1.txt";
    reader.open(base_path + filename);
    for (int i = 0; i < probs->numel(); i++) {
      reader >> probs_data[i];
    }
    LOG(INFO) << "Read Probs data." << probs_data[0];
    reader.close();
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("generate_proposals");

    op_desc->SetInput("Scores", {Scores_});
    op_desc->SetInput("BboxDeltas", {BboxDeltas_});
    op_desc->SetInput("ImInfo", {ImInfo_});
    op_desc->SetInput("Anchors", {Anchors_});
    op_desc->SetInput("Variances", {Variances_});

    op_desc->SetAttr("pre_nms_topN", pre_nms_topN_);
    op_desc->SetAttr("post_nms_topN", post_nms_topN_);
    op_desc->SetAttr("nms_thresh", nms_thresh_);
    op_desc->SetAttr("min_size", min_size_);
    op_desc->SetAttr("eta", eta_);

    op_desc->SetOutput("RpnRois", {RpnRois_});
    op_desc->SetOutput("RpnRoiProbs", {RpnRoiProbs_});
  }

  void PrepareData() override {
    std::string base_path = "/data/local/tmp/data_files/";
    std::string filename;
    DDim dims;
    std::vector<float> datas;
    std::ifstream reader;
    // Scores
    filename = "result_rpn_cls_prob.tmp_0.txt";
    dims = DDim(std::vector<int64_t>({2, 15, 84, 50}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read Scores data." << datas[0];
    reader.close();
    SetCommonTensor(Scores_, dims, datas.data());

    // BboxDeltas
    filename = "result_rpn_bbox_pred.tmp_1.txt";
    dims = DDim(std::vector<int64_t>({2, 60, 84, 50}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read BboxDeltas  data." << datas[0];
    reader.close();
    reader.close();
    SetCommonTensor(BboxDeltas_, dims, datas.data());

    // ImInfo
    filename = "result_im_info.txt";
    dims = DDim(std::vector<int64_t>({2, 3}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read ImInfo  data." << datas[0];
    reader.close();
    SetCommonTensor(ImInfo_, dims, datas.data());

    // Anchors
    filename = "result_anchor_generator_0.tmp_0.txt";
    dims = DDim(std::vector<int64_t>({84, 50, 15, 4}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read Anchors  data." << datas[0];
    reader.close();
    SetCommonTensor(Anchors_, dims, datas.data());

    // Variances
    filename = "result_anchor_generator_0.tmp_1.txt";
    dims = DDim(std::vector<int64_t>({84, 50, 15, 4}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read Variances  data." << datas[0];
    reader.close();
    SetCommonTensor(Variances_, dims, datas.data());
  }
};

TEST(GenerateProposals, precision) {
  // The unit test for generate_proposals needs the params,
  // which is obtained by runing model by paddle.
  LOG(INFO) << "test generate proposals op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  std::unique_ptr<arena::TestCase> tester(
      new GenerateProposalsComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
