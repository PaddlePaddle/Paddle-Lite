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
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

class CrfDecodingComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string emission_ = "Emission";
  std::string transition_ = "Transition";
  std::string output_ = "ViterbiPath";

 public:
  CrfDecodingComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize({5, 1});
    LoD out_lod;
    out_lod.push_back({0, 2, 5});
    out->set_lod(out_lod);

    std::vector<int64_t> data = {0, 1, 0, 2, 2};
    auto* out_data = out->mutable_data<int64_t>();
    for (int i = 0; i < data.size(); i++) {
      out_data[i] = data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("crf_decoding");
    op_desc->SetInput("Emission", {emission_});
    op_desc->SetInput("Transition", {transition_});
    op_desc->SetOutput("ViterbiPath", {output_});
  }

  void PrepareData() override {
    std::vector<float> emission_data = {0.39293837,
                                        -0.42772133,
                                        -0.54629709,
                                        0.10262954,
                                        0.43893794,
                                        -0.15378708,
                                        0.9615284,
                                        0.36965948,
                                        -0.0381362,
                                        -0.21576496,
                                        -0.31364397,
                                        0.45809941};
    LoD lod;
    lod.push_back({0, 2, 5});
    SetCommonTensor(emission_, DDim({5, 3}), emission_data.data(), lod);

    std::vector<float> transition_data = {0.2379954057320357,
                                          -0.3175082695465,
                                          -0.32454824385250747,
                                          0.03155137384183837,
                                          0.03182758709686606,
                                          0.13440095855132106,
                                          0.34943179407778957,
                                          0.22445532486063524,
                                          0.11102351067758287,
                                          0.22244338257022156,
                                          -0.1770410861468218,
                                          -0.1382113443776859,
                                          -0.2717367691210444,
                                          -0.20628595361117064,
                                          0.13097612385448776};
    SetCommonTensor(transition_, DDim({5, 3}), transition_data.data());
  }
};

TEST(CrfDecoding, arm_precision) {
  LOG(INFO) << "test crf_decoding op";
#ifdef LITE_WITH_X86
  Place place(TARGET(kHost));
  std::unique_ptr<arena::TestCase> tester(
      new CrfDecodingComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif

#ifdef LITE_WITH_ARM
  Place place(TARGET(kHost));
  std::unique_ptr<arena::TestCase> tester(
      new CrfDecodingComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
