/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_mul_pe.hpp"
#include "lite/backends/fpga/KD/pes/fully_connected_pe.hpp"
#include "lite/backends/fpga/KD/pes/relu_pe.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/arm/math/sgemm.h"

#include "lite/backends/arm/math/funcs.h"
#include "lite/api/paddle_place.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace zynqmp {

struct GRUTensors {
  Tensor* gate;
  Tensor* pre_output;
  Tensor* output;
  Tensor* reset_output;
};

class GRUPE : public PE {
 public:


  bool init() {
    // Tensor* output = param_.output;
    // output->setAligned(true);
    // output->setDataLocation(Device);
    return true;
  }

  void apply() {
    auto hidden = param_.hidden;
    // auto hidden_dims = hidden->dims();
    int frame_size = hidden->shape().channel();
    
    zynqmp::Shape hidden_shape{zynqmp::NCHW, {1, frame_size, 1, 1}};
    float16* prev_hidden_data = prev_hidden_.mutableData<float16>(zynqmp::FP16, hidden_shape);
    // set previous hidden data to 0;
    memset(prev_hidden_data, 0, hidden_shape.numel() * sizeof(float16));

    // copy 2/3 weight from param.weight;
    zynqmp::Shape weight_shape{zynqmp::NC, {frame_size, frame_size * 2}};
    float* weight_data = weight_.mutableData<float>(zynqmp::FP32, weight_shape);
    memset(weight_data, 0, weight_shape.numel() * sizeof(float));
    weight_data = weight_.mutableData<float>(zynqmp::FP32, weight_shape);
    memcpy(weight_data, param_.weight->data<float>(), weight_shape.numel() * sizeof(float));

    Shape gate_shape(zynqmp::NC, {1, frame_size * 2});
    gate_ping_.mutableData<void>(FP32, gate_shape);
    gate_pong_.mutableData<void>(FP16, gate_shape);

    zynqmp::FullyConnectedParam& pre_out_param = pre_out_pe_.param();
    pre_out_param.input = &prev_hidden_;
    pre_out_param.output = &gate_pong_;
    pre_out_param.filter = &weight_;
    pre_out_param.bias = &gate_ping_;
    pre_out_pe_.init();
    pre_out_pe_.apply();

    // // ============= C
    // ElementwiseAddParam& bias_add_param = bias_ew_pe_.param();
    // bias_add_param.inputs = {&pre_output_, &pre_input_};
    // bias_add_param.output = &pre_input_;
    // bias_ew_pe_.init();
    // bias_ew_pe_.apply();
    // // ====================

    // Shape state_weight_shape(NC,{frame_size, frame_size});
    // float* state_weight_data = state_weight_.mutableData<float>(FP32, state_weight_shape);
    // memcpy(state_weight_data, weight_data + 2 * frame_size * frame_size, 
    //   state_weight_shape.numel() * sizeof(float));
    // FullyConnectedParam& reset_out_param = reset_out_pe_.param();
    // reset_out_param.input = &prev_hidden;
    // reset_out_param.output = &gate_ping;
    // reset_out_param.filter = &state_weight_;

    // // ============== unit reset;
    // update_gate_.mutableData<void>(FP16, pre_input_shape);
    // InputParam& relu_param = update_relu_pe_.param(); 
    // relu_param.input = &tempTensor;
    // relu_param.output = &update_gate_;
    // update_relu_pe_.init();
    // update_relu_pe_.apply();

    
    reset_gate_.mutableData<void>(FP16, hidden_shape);
    prev_hidden_.mutableData<void>(FP16, hidden_shape);
    reset_hidden_.mutableData<void>(FP16, hidden_shape);
    // InputParam& reset_param = reset_relu_pe_.param();
    // reset_param.input = &tempTensor;
    // reset_param.output = &reset_gate_;
    // reset_relu_pe_.init();
    // reset_relu_pe_.apply();

    // float16* prev_data = prev_.mutableData<float16>(FP16, pre_input_shape);
    // memset(prev_data, 0, (pre_input_shape.numel() + 32) * sizeof(float16)); // TODO
    // reset_hidden_prev_.mutableData<float16>(FP16, pre_input_shape);

    ElementwiseMulParam& mul_param = mul_pe_.param();
    mul_param.inputs = {&reset_gate_, &prev_hidden_};
    mul_param.output = &reset_hidden_;
    mul_pe_.init();
    mul_pe_.apply();
    // ============== 

  }

  bool dispatch() {
    return true;
  }

  void gru_unit_reset_act(const lite_api::ActivationType active_gate, GRUTensors& value,
                int frame_size, int batch_size) {

    int stride_update = 3 * frame_size;
    int stride_cell_state = 3 * frame_size;
    int stride_hidden_prev = frame_size;
    int stride_hidden = frame_size;

    // Tensor* gate = value.gate;
    // value.gate->saveToFile("value_input.txt");

    float* update_gate_data = gate_ping_.data<float>();
    float* reset_gate_data = update_gate_data + frame_size;

    for (int b = 0; b < batch_size; b++) {
      // memcpy(tempTensor.data<void>(), reset_gate_data, gate->shape().numel() * sizeof(float));
      // tempTensor.flush();

      Tensor tmp;
      Shape s(NC, {1, frame_size}); //TODO
      float* tmp_data = tmp.mutableData<float>(FP32, s);

      for (int i = 0; i < frame_size; i++) {
        // f(x) = x / (1 + abs(x))?
        update_gate_data[i] = lite::arm::math::active_f32<lite_api::ActivationType::kSigmoid>(update_gate_data[i]);
        reset_gate_data[i] = lite::arm::math::active_f32<lite_api::ActivationType::kSigmoid>(reset_gate_data[i]);
      }
      memcpy(tmp_data, reset_gate_data, frame_size * sizeof(float));
      tmp.flush();
      reset_gate_.copyFrom(&tmp);

      // reset_gate_.copyFrom(&tempTensor);
      Tensor* hidden_prev = value.pre_output;
      if (hidden_prev) {
        // memcpy(prev_data, )
        // TODO change to pre_out;
        prev_hidden_.copyFrom(value.pre_output);
        prev_hidden_.saveToFile("prev_.txt");
      }

      // // 4.0 reset_date * hidden_prev;
      // // reset_hidden_prev[i] = reset_gate[i] * prev;
      mul_pe_.dispatch();
      reset_hidden_.saveToFile("reset_hidden_.txt");
      update_gate_data += stride_update;
      reset_gate_data += stride_update;

      // reset_hidden_prev += stride_hidden;// TODO
    }
  }

  void gru_unit_out_act(const lite_api::ActivationType active_node, bool origin_mode,
                GRUTensors& value, int frame_size, int batch_size) {

    // int stride_update = 3 * frame_size;
    // int stride_cell_state = 3 * frame_size;
    // int stride_hidden_prev = frame_size;
    // int stride_hidden = frame_size;

    // Tensor* hidden = value.output_value;
    // float* hidden_prev = nullptr;
    // if (hidden) {
    //   hidden_prev = hidden->data<float>();
    // }

    // float* cell_state = value.gate->data<float>() + 2 * frame_size;

    // float* updata_gate = value.gate->data<float>();
    // // float* reset_gate_data = update_gate_data + frame_size;

    // float prev = 0.0f;
    // for (int b = 0; b < batch_size; ++b) {
    //   if (origin_mode) {
    //     // for (int i = 0; i < frame_size; i++) {
    //     //   float prev = 0;
    //     //   if (hidden_prev) {
    //     //     prev = hidden_prev[i];
    //     //   }
    //     //   cell_state[i] = lite::arm::math::active_f32<kSigmoid>(cell_state[i]);
    //     //   hidden[i] =
    //     //       cell_state[i] * (1.f - updata_gate[i]) + updata_gate[i] * prev;
    //     // }
    //   } else {
    //     for (int i = 0; i < frame_size; ++i) {
    //       cell_state[i] = lite::arm::math::active_f32<lite_api::ActivationType::kRelu>(cell_state[i]);
    //       if (hidden_prev) {
    //        prev = hidden_prev[i];
    //       }
    //       float hidden_value =
    //         prev * (1.f - updata_gate[i]) + updata_gate[i] * cell_state[i];
    //       hidden_prev[i] = hidden_value;
    //       std::cout << "hidden_value::" << hidden_value << std::endl;
    //     }
    //   }
    //   updata_gate += stride_update;
    //   cell_state += stride_cell_state;
    //   hidden_prev += frame_size;
    // }
  }

  void copy_input(GRUTensors& value) {
    float max = find_max(*(value.gate));
    gate_ping_.mutableData<void>(FP32, value.gate->shape());
    gate_ping_.copyFrom(value.gate);
    // TODO update input pointer?


    // gate_.readFromFile("input/in.txt");
    // // pre_input_.saveToFile("pppp_in.txt");
    // gate_.scale()[0] = max / 127;
    // gate_.scale()[1] = 127 / max;
    // gate_.printScale("pre_input_");


    // gate_.saveToFile("pre_input_.txt");

    // pre_out_pe_.dispatch();

    // pre_output_.saveToFile("pp_out.txt");
  }

  void GRUCOmpute(GRUTensors& value,
                      int frame_size,
                      int batch_size,
                      const lite_api::ActivationType active_node,
                      const lite_api::ActivationType active_gate,
                      bool origin_mode) {
    copy_input(value);

    if (value.pre_output) {
      // copy by batch;
      pre_out_pe_.dispatch();
      gate_ping_.copyFrom(&gate_pong_);
    }

    gru_unit_reset_act(active_gate, value, frame_size, batch_size);

    // if (value.pre_output) {
    //   // state weight;
    //   reset_out_pe_.dispatch();
    // }
    // gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
  }

  GRUParam& param() { return param_; }

  // Tensor* preOutput() {
  //   return &pre_output_;
  // }

  // Tensor* gate() {
  //   return &gate_;
  // }

  Tensor* updateGate() {
    return &update_gate_;
  }

  Tensor* resetGate() {
    return &reset_gate_;
  }

 private:
  GRUParam param_;
  zynqmp::Tensor gate_ping_;
  zynqmp::Tensor gate_pong_;
  zynqmp::Tensor bias_;
  zynqmp::Tensor weight_;
  zynqmp::Tensor state_weight_;
  // =================================
  zynqmp::Tensor update_gate_;
  zynqmp::Tensor reset_gate_;
  zynqmp::Tensor cell_state_;
  zynqmp::Tensor prev_hidden_;
  zynqmp::Tensor reset_hidden_;

  Tensor tempTensor;
  // =================================

  ReluPE update_relu_pe_;
  ReluPE reset_relu_pe_;
  zynqmp::ElementwiseMulPE mul_pe_;
  zynqmp::FullyConnectedPE pre_out_pe_;
  zynqmp::FullyConnectedPE reset_out_pe_;

  zynqmp::ElementwiseAddPE bias_ew_pe_;
};

}  // namespace zynqmp
}  // namespace paddle
