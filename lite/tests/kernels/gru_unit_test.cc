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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

static float sigmoid(float a) { return 1.f / (1.f + exp(-a)); }

static float tanh(float a) {
  float tmp = -2.f * a;
  return 2.f / (1.f + exp(tmp)) - 1.f;
}

static float relu(float a) { return a > 0.f ? a : 0.f; }

static float identity(float a) { return a; }

typedef float (*act_func)(float a);

void gru_add_with_bias(const float* din,
                       const float* bias,
                       bool flag_bias,
                       float* dout,
                       int batch,
                       int size) {
  for (int i = 0; i < batch; ++i) {
    auto din_batch = din + i * size;
    auto dout_batch = dout + i * size;
    if (flag_bias) {
      for (int j = 0; j < size; ++j) {
        dout_batch[j] = din_batch[j] + bias[j];
      }
    } else {
      memcpy(dout_batch, din_batch, size * sizeof(float));
    }
  }
}

void gru_unit_reset_act_host(act_func act,
                             float* updata_gate,
                             int stride_update,
                             float* reset_gate,
                             int stride_reset,
                             const float* hidden_prev,
                             int stride_hidden_prev,
                             float* reset_hidden_prev,
                             int stride_reset_hidden_prev,
                             int frame_size,
                             int batch_size) {
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < frame_size; ++i) {
      updata_gate[i] = act(updata_gate[i]);
      reset_gate[i] = act(reset_gate[i]);
      reset_hidden_prev[i] = reset_gate[i] * hidden_prev[i];
    }
    updata_gate += stride_update;
    reset_gate += stride_reset;
    hidden_prev += stride_hidden_prev;
    reset_hidden_prev += stride_reset_hidden_prev;
  }
}

void gru_unit_out_act_host(act_func act,
                           bool origin_mode,
                           const float* updata_gate,
                           int stride_update,
                           float* cell_state,
                           int stride_cell_state,
                           const float* hidden_prev,
                           int stride_hidden_prev,
                           float* hidden,
                           int stride_hidden,
                           int frame_size,
                           int batch_size) {
  for (int b = 0; b < batch_size; ++b) {
    if (origin_mode) {
      for (int i = 0; i < frame_size; ++i) {
        cell_state[i] = act(cell_state[i]);
        hidden[i] = cell_state[i] * (1.f - updata_gate[i]) +
                    updata_gate[i] * hidden_prev[i];
      }
    } else {
      for (int i = 0; i < frame_size; ++i) {
        cell_state[i] = act(cell_state[i]);
        hidden[i] = hidden_prev[i] * (1.f - updata_gate[i]) +
                    updata_gate[i] * cell_state[i];
      }
    }
    updata_gate += stride_update;
    cell_state += stride_cell_state;
    hidden_prev += stride_hidden_prev;
    hidden += stride_hidden;
  }
}

void gru_unit_basic(const Tensor* input,
                    const Tensor* hidden_prev,
                    const Tensor* weights,
                    const Tensor* bias,
                    Tensor* gate,
                    Tensor* reset_hidden_prev,
                    Tensor* hidden,
                    int act_gate,
                    int act,
                    bool origin_mode) {
  auto batch_size = input->dims()[0];
  auto frame_size = hidden_prev->dims()[1];
  auto input_data = input->data<float>();
  auto hidden_prev_data = hidden_prev->data<float>();
  auto weight_data = weights->data<float>();

  auto gate_data = gate->mutable_data<float>();
  auto reset_hidden_prev_data = reset_hidden_prev->mutable_data<float>();
  auto hidden_data = hidden->mutable_data<float>();

  act_func act_gate_func{nullptr};
  act_func act_func{nullptr};
  switch (act_gate) {
    case 0:
      act_gate_func = identity;
      break;
    case 1:
      act_gate_func = sigmoid;
      break;
    case 2:
      act_gate_func = tanh;
      break;
    case 3:
      act_gate_func = relu;
      break;
    default:
      break;
  }
  switch (act) {
    case 0:
      act_func = identity;
      break;
    case 1:
      act_func = sigmoid;
      break;
    case 2:
      act_func = tanh;
      break;
    case 3:
      act_func = relu;
      break;
    default:
      break;
  }

  const float* bias_data = nullptr;
  bool flag_bias = false;
  if (bias) {
    bias_data = bias->data<float>();
    flag_bias = true;
  }
  gru_add_with_bias(
      input_data, bias_data, flag_bias, gate_data, batch_size, frame_size * 3);
  basic_gemm(false,
             false,
             batch_size,
             2 * frame_size,
             frame_size,
             1.f,
             hidden_prev_data,
             frame_size,
             weight_data,
             frame_size * 2,
             1.f,
             gate_data,
             frame_size * 3,
             (const float*)nullptr,
             false,
             false);

  gru_unit_reset_act_host(act_gate_func,
                          gate_data,
                          3 * frame_size,
                          gate_data + frame_size,
                          3 * frame_size,
                          hidden_prev_data,
                          frame_size,
                          reset_hidden_prev_data,
                          frame_size,
                          frame_size,
                          batch_size);

  basic_gemm(false,
             false,
             batch_size,
             frame_size,
             frame_size,
             1.f,
             reset_hidden_prev_data,
             frame_size,
             weight_data + 2 * frame_size * frame_size,
             frame_size,
             1.f,
             gate_data + frame_size * 2,
             frame_size * 3,
             bias_data,
             false,
             false);

  gru_unit_out_act_host(act_func,
                        origin_mode,
                        gate_data,
                        3 * frame_size,
                        gate_data + 2 * frame_size,
                        3 * frame_size,
                        hidden_prev_data,
                        frame_size,
                        hidden_data,
                        frame_size,
                        frame_size,
                        batch_size);
}

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

  // 0: indentity; 1: sigmoid; 2: tanh; 3: relu
  int gate_activation_{1};
  int activation_{2};
  bool origin_mode_{false};
  DDim dims_{{16, 256 * 3}};

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
        dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto input = scope->FindTensor(input_);
    auto hidden_prev = scope->FindTensor(hidden_prev_);
    auto weights = scope->FindTensor(weight_);
    auto bias = scope->FindTensor(bias_);

    auto batch_size = input->dims()[0];
    auto frame_size = hidden_prev->dims()[1];

    auto hidden = scope->NewTensor(hidden_);
    auto reset_hidden_prev = scope->NewTensor(reset_hidden_prev_);
    auto gate = scope->NewTensor(gate_);

    CHECK(hidden);
    CHECK(reset_hidden_prev);
    CHECK(gate);
    hidden->Resize(lite::DDim({batch_size, frame_size}));
    reset_hidden_prev->Resize(lite::DDim({batch_size, frame_size}));
    gate->Resize(lite::DDim({batch_size, 3 * frame_size}));

    gru_unit_basic(input,
                   hidden_prev,
                   weights,
                   bias,
                   gate,
                   reset_hidden_prev,
                   hidden,
                   gate_activation_,
                   activation_,
                   origin_mode_);
  }

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
    fill_data_rand(data.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, data.data());

    // set hidden_prev data
    data.resize(hpdim.production());
    fill_data_rand(data.data(), -1.f, 1.f, hpdim.production());
    SetCommonTensor(hidden_prev_, hpdim, data.data());

    // set weight data
    data.resize(wdim.production());
    fill_data_rand(data.data(), -1.f, 1.f, wdim.production());
    SetCommonTensor(weight_, wdim, data.data());

    // set bias data
    data.resize(bdim.production());
    fill_data_rand(data.data(), -1.f, 1.f, bdim.production());
    SetCommonTensor(bias_, bdim, data.data());
  }
};

void test_gru_unit(Place place) {
  DDimLite dims{{8, 16 * 3}};
  std::unique_ptr<arena::TestCase> tester(new GRUUnitTester(
      place, "def", 1 /* sigomoid */, 2 /* tanh */, false, dims));
#ifdef LITE_WITH_ARM
  auto& ctx = tester->context()->template As<ARMContext>();
  ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 1);
#endif
  arena::Arena arena(std::move(tester), place, 1e-4);
  arena.TestPrecision();
}

TEST(GRUUnit, precision) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  test_gru_unit(place);
}

}  // namespace lite
}  // namespace paddle
