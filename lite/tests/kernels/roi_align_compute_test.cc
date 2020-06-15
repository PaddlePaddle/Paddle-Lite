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

class RoiAlignComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string rois_ = "ROIs";
  std::string out_ = "Out";
  float spatial_scale_ = 0.0625;
  int pooled_height_ = 14;
  int pooled_width_ = 14;
  int sampling_ratio_ = 0;

 public:
  RoiAlignComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(std::vector<int64_t>({304, 1024, 14, 14}));
    /*
    std::vector<uint64_t> lod0({0, 152, 304});
    LoD lod;
    lod.push_back(lod0);
    probs->set_lod(lod);
    */

    auto* out_data = out->mutable_data<float>();

    std::string base_path = "/data/local/tmp/roi_align_datas/";
    std::string filename;
    std::ifstream reader;
    // out
    filename = "result_roi_align_0.tmp_0.txt";
    reader.open(base_path + filename);
    LOG(INFO) << "Start read out data";
    for (int i = 0; i < out->numel(); i++) {
      reader >> out_data[i];
    }
    LOG(INFO) << "Read out data. " << out_data[0] << " "
              << out_data[out->numel() - 1];
    reader.close();
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("roi_align");

    op_desc->SetInput("X", {x_});
    op_desc->SetInput("ROIs", {rois_});

    op_desc->SetAttr("spatial_scale", spatial_scale_);
    op_desc->SetAttr("pooled_height", pooled_height_);
    op_desc->SetAttr("pooled_width", pooled_width_);
    op_desc->SetAttr("sampling_ratio", sampling_ratio_);

    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::string base_path = "/data/local/tmp/roi_align_datas/";
    std::string filename;
    DDim dims;
    std::vector<float> datas;
    std::ifstream reader;
    // x
    filename = "result_res4f.add.output.5.tmp_0.txt";
    dims = DDim(std::vector<int64_t>({2, 1024, 84, 50}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read x data. " << datas[0] << " " << datas.back();
    reader.close();
    SetCommonTensor(x_, dims, datas.data());

    // rois
    filename = "result_generate_proposals_0.tmp_0.txt";
    dims = DDim(std::vector<int64_t>({304, 4}));
    datas.resize(dims.production());
    reader.open(base_path + filename);
    for (int i = 0; i < dims.production(); i++) {
      reader >> datas[i];
    }
    LOG(INFO) << "Read rois  data. " << datas[0] << " " << datas.back();
    reader.close();
    SetCommonTensor(rois_, dims, datas.data());

    auto rois_tensor = baseline_scope()->FindMutableTensor(rois_);
    std::vector<uint64_t> lod0({0, 152, 304});
    LoD lod;
    lod.push_back(lod0);
    rois_tensor->set_lod(lod);
  }
};

TEST(RoiAlign, precision) {
  // The unit test for roi_align needs the params,
  // which is obtained by runing model by paddle.
  LOG(INFO) << "test roi align op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  std::unique_ptr<arena::TestCase> tester(
      new RoiAlignComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
