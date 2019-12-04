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

#pragma once

#include "lite/backends/arm/math/gru_utils.h"

namespace paddle {
namespace lite {
namespace fpga {

// inline void gru_unit_reset_act(lite_api::ActivationType act_type,
//                                GRUMetaValue<float> value,
//                                int frame_size,
//                                int batch_size) {
//   auto updata_gate = value.gate_value;
//   auto reset_gate = value.gate_value + frame_size;
//   auto hidden_prev = value.prev_out_value;
//   auto reset_hidden_prev = value.reset_output_value;
//   int stride_update = 3 * frame_size;
//   int stride_reset = 3 * frame_size;
//   int stride_hidden_prev = frame_size;
//   int stride_reset_hidden_prev = frame_size;

//   if (act_type == kRelu) {
    
//   }
// }

// void gru_compute(arm::math::GRUMetaValue<float> value,
//                       int frame_size,
//                       int batch_size,
//                       const lite_api::ActivationType active_node,
//                       const lite_api::ActivationType active_gate,
//                       bool origin_mode) {

// 	std::cout << " =================== gru gru_compute =================== \n";
// 	// exit(-1);

// 	// sgemm(bool is_transA,
//  //           bool is_transB,
//  //           int M,
//  //           int N,
//  //           int K,
//  //           float alpha,
//  //           const float* A,
//  //           int lda,
//  //           const float* B,
//  //           int ldb,
//  //           float beta,
//  //           float* C,
//  //           int ldc,
//  //           const float* bias,
//  //           bool is_bias,
//  //           bool is_relu,
//  //           ARMContext* ctx);

// 	// sgemm for fc;
//     // lite::arm::math::sgemm(false,
//     //                    false,
//     //                    m_,// batch;
//     //                    n_,// filter num;
//     //                    k_,// input_channel;
//     //                    1.f,
//     //                    i_data,// input data;
//     //                    k_,
//     //                    w_data,// weight data;
//     //                    n_,
//     //                    0.f,//beta;
//     //                    o_data,// out data;
//     //                    n_,
//     //                    b_data,// bias;
//     //                    false,
//     //                    false,
//     //                    &ctx);

// 	// C := alpha*op( A )*op( B ) + beta*C,

// 	if (value.prev_out_value) {
//       // sgemm(false, // is_transA
//       //       false, // is_transB
//       //       batch_size, // M specifies  the number  of rows  of the  matrix
//       //       frame_size * 2, // N specifies the number  of columns of the matrix
//       //       frame_size, // K
//       //       1.f,  // alpha
//       //       value.prev_out_value, // float* A,
//       //       frame_size, // lda
//       //       value.gate_weight, // float* B,
//       //       frame_size * 2, // ldb
//       //       1.f, // beta
//       //       value.gate_value, // C*
//       //       frame_size * 3, // ldc
//       //       nullptr, // bias
//       //       false, // is_bias
//       //       false, // is_relu
//       //       ctx); // context

//     }
//     // gru_unit_reset_act(active_gate, value, frame_size, batch_size);

//     if (value.prev_out_value) {
//       // sgemm(false,
//       //       false,
//       //       batch_size,
//       //       frame_size,
//       //       frame_size,
//       //       1.f,
//       //       value.reset_output_value,
//       //       frame_size,
//       //       value.state_weight,
//       //       frame_size,
//       //       1.f,
//       //       value.gate_value + frame_size * 2,
//       //       frame_size * 3,
//       //       nullptr,
//       //       false,
//       //       false,
//       //       ctx);
//     }

//     // gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
// }


}
}
}