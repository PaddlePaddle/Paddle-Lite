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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/kernels/fill_data.h"

namespace paddle {
namespace lite {

class GRUUnitTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "input";
  std::string hidden_prev_ = "hidden_prev";
  std::string weight_ = "weight";
  std::string bias_ = "bias";
  std::string gate_ = "gate";
  std::string reset_hidden_prev_ = "reset_hidden_prev";
  std::string hidden_ = "hidden";

  DDim dims_{{16, 256 * 3}};
  // 0: indentity; 1: sigmoid; 2: tanh; 3: relu
  int gate_activation_{1};
  int activation_{2};
  bool origin_mode_{false};

 public:
  GRUUnitTester(const Place& place,
                const std::string& alias,
                int gate_activation,
                int activation,
                bool origin_mode,
                DDim dims)
      : TestCase(place, alias),
        gate_activation_(gate_activation),
        activation_(activation),
        origin_mode_(origin_mode),
        dims_(dims) {
    Prepare();
  }

  void RunBaseline(Scope* scope) override {}

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("gru_unit");
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("HiddenPrev", {hidden_prev_});
    op_desc->SetInput("Weight", {weight_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetOutput("Gate", {gate_});
    op_desc->SetOutput("ResetHiddenPrev", {reset_hidden_prev_});
    op_desc->SetOutput("Hidden", {hidden_});

    op_desc->SetAttr("gate_activation", gate_activation_);
    op_desc->SetAttr("activation", activation_);
    op_desc->SetAttr("origin_mode", origin_mode_);
  }

  void PrepareData() override {
    int64_t batch_size = dims_[0];
    int64_t frame_size = dims_[1] / 3;
    DDim wdim{{frame_size, frame_size * 3}};
    DDim bdim{{1, frame_size * 3}};
    DDim hpdim{{batch_size, frame_size}};

    // set input data
    std::vector<float> data(dims_.production());
    fill_data_rand(data.data(), 0.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, data.data());

    // set hidden_prev data
    data.resize(hpdim.production());
    fill_data_rand(data.data(), 0.f, 1.f, hpdim.production());
    SetCommonTensor(hidden_prev_, hpdim, data.data());

    // set weight data
    data.resize(wdim.production());
    fill_data_rand(data.data(), 0.f, 1.f, wdim.production());
    SetCommonTensor(weight_, wdim, data.data());

    // set bias data
    data.resize(bdim.production());
    fill_data_rand(data.data(), 0.f, 1.f, bdim.production());
    SetCommonTensor(bias_, bdim, data.data());
  }
};

void test_gru_unit(Place place) {
  DDimLite dims{{16, 256 * 3}};
  std::unique_ptr<arena::TestCase> tester(new GRUUnitTester(
      place, "def", 1 /* sigomoid */, 2 /* tanh */, false, dims));
  auto& ctx = tester->context()->template As<ARMContext>();
  ctx.SetRunMode(LITE_POWER_HIGH, 1);
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

TEST(GRUUnit, precision) {
#ifdef LITE_WITH_ARM
  //  DeviceInfo::Init();
  Place place(TARGET(kARM));
#else
  Place place(TARGET(kHost));
#endif
  test_gru_unit(place);
}

}  // namespace lite
}  // namespace paddle
