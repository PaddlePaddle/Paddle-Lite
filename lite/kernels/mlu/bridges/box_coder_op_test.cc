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

#include "lite/operators/box_coder_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/core/op_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void ToFile(Tensor *tensor, std::string file_name) {
  int count = tensor->dims().production();
  auto data = tensor->mutable_data<float>();
  std::ostringstream outs;
  for (size_t i = 0; i < count; i++) {
    outs << data[i] << std::endl;
  }
  std::ofstream of;
  of.open(file_name, std::ios::out);
  of << outs.str();
  of.close();
}

inline std::string BoxCodeTypeToStr(cnmlBoxCodeType_t code_type) {
  if (code_type == cnmlBoxCodeType_t::Encode) {
    return "encode_center_size";
  } else if (code_type == cnmlBoxCodeType_t::Decode) {
    return "decode_center_size";
  } else {
    CHECK(false);
  }
}

inline cnmlBoxCodeType_t GetBoxCodeType(const std::string &type) {
  if (type == "encode_center_size") {
    return cnmlBoxCodeType_t::Encode;
  } else if (type == "decode_center_size") {
    return cnmlBoxCodeType_t::Decode;
  } else {
    CHECK(false);
  }
}

void EncodeCenterSize(float *target_box_data,
                      float *prior_box_data,
                      float *prior_box_var_data,
                      std::vector<int64_t> target_box_shape,
                      std::vector<int64_t> prior_box_shape,
                      std::vector<int64_t> prior_box_var_shape,
                      const bool normalized,
                      const std::vector<float> variance,
                      float *output) {
  int64_t row = target_box_shape[0];
  int64_t col = prior_box_shape[0];
  int64_t len = prior_box_shape[1];

  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      size_t offset = i * col * len + j * len;
      float prior_box_width = prior_box_data[j * len + 2] -
                              prior_box_data[j * len] + (normalized == false);
      float prior_box_height = prior_box_data[j * len + 3] -
                               prior_box_data[j * len + 1] +
                               (normalized == false);
      float prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[j * len + 1] + prior_box_height / 2;

      float target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      float target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      float target_box_width = target_box_data[i * len + 2] -
                               target_box_data[i * len] + (normalized == false);
      float target_box_height = target_box_data[i * len + 3] -
                                target_box_data[i * len + 1] +
                                (normalized == false);

      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width));
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height));
    }
  }

  if (prior_box_var_data) {
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          size_t offset = i * col * len + j * len;
          int prior_var_offset = j * len;
          output[offset + k] /= prior_box_var_data[prior_var_offset + k];
        }
      }
    }
  } else if (!(variance.empty())) {
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          size_t offset = i * col * len + j * len;
          output[offset + k] /= static_cast<float>(variance[k]);
        }
      }
    }
  }
}

template <int axis, int var_size>
void DecodeCenterSize(float *target_box_data,
                      float *prior_box_data,
                      float *prior_box_var_data,
                      std::vector<int64_t> target_box_shape,
                      std::vector<int64_t> prior_box_shape,
                      std::vector<int64_t> prior_box_var_shape,
                      const bool normalized,
                      std::vector<float> variance,
                      float *output) {
  int64_t row = target_box_shape[0];
  int64_t col = target_box_shape[1];
  int64_t len = target_box_shape[2];

  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      float var_data[4] = {1., 1., 1., 1.};
      float *var_ptr = var_data;
      size_t offset = i * col * len + j * len;
      int prior_box_offset = axis == 0 ? j * len : i * len;

      float prior_box_width = prior_box_data[prior_box_offset + 2] -
                              prior_box_data[prior_box_offset] +
                              (normalized == false);
      float prior_box_height = prior_box_data[prior_box_offset + 3] -
                               prior_box_data[prior_box_offset + 1] +
                               (normalized == false);
      float prior_box_center_x =
          prior_box_data[prior_box_offset] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[prior_box_offset + 1] + prior_box_height / 2;

      float target_box_center_x = 0, target_box_center_y = 0;
      float target_box_width = 0, target_box_height = 0;
      int prior_var_offset = axis == 0 ? j * len : i * len;
      if (var_size == 2) {
        std::memcpy(
            var_ptr, prior_box_var_data + prior_var_offset, 4 * sizeof(float));
      } else if (var_size == 1) {
        var_ptr = reinterpret_cast<float *>(variance.data());
      }
      float box_var_x = *var_ptr;
      float box_var_y = *(var_ptr + 1);
      float box_var_w = *(var_ptr + 2);
      float box_var_h = *(var_ptr + 3);

      target_box_center_x =
          box_var_x * target_box_data[offset] * prior_box_width +
          prior_box_center_x;
      target_box_center_y =
          box_var_y * target_box_data[offset + 1] * prior_box_height +
          prior_box_center_y;
      target_box_width =
          std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width;
      target_box_height =
          std::exp(box_var_h * target_box_data[offset + 3]) * prior_box_height;

      output[offset] = target_box_center_x - target_box_width / 2;
      output[offset + 1] = target_box_center_y - target_box_height / 2;
      output[offset + 2] =
          target_box_center_x + target_box_width / 2 - (normalized == false);
      output[offset + 3] =
          target_box_center_y + target_box_height / 2 - (normalized == false);
    }
  }
}

void Compute(cnmlBoxCodeType_t code_type,
             lite::Tensor *prior_box,
             lite::Tensor *target_box,
             lite::Tensor *box_var,
             lite::Tensor *output_box,
             std::vector<float> variance,
             bool normalized,
             int axis) {
  // BoxCodeType code_type = BoxCodeType::kDecodeCenterSize;
  // std::vector<int> prior_box_shape = {512, 4};
  // std::vector<int> prior_box_var_shape = prior_box_shape;

  // std::vector<int> target_box_shape;
  // std::vector<int> output_shape;
  // if (code_type == BoxCodeType::kEncodeCenterSize) {
  //   target_box_shape = {81, 4};
  //   output_shape = {81, 512, 4};
  // } else {
  //   target_box_shape = {81, 512, 4};
  //   output_shape = {81, 512, 4};
  // }

  auto *prior_box_data = prior_box->mutable_data<float>();
  auto *prior_box_var_data = box_var->mutable_data<float>();
  auto *target_box_data = target_box->mutable_data<float>();
  auto *output_data = output_box->mutable_data<float>();

  auto target_box_shape = target_box->dims().Vectorize();
  auto prior_box_shape = prior_box->dims().Vectorize();
  auto prior_box_var_shape = box_var->dims().Vectorize();
  if (code_type == cnmlBoxCodeType_t::Encode) {
    EncodeCenterSize(target_box_data,
                     prior_box_data,
                     prior_box_var_data,
                     target_box_shape,
                     prior_box_shape,
                     prior_box_var_shape,
                     normalized,
                     variance,
                     output_data);
  } else if (code_type == cnmlBoxCodeType_t::Decode) {
    if (prior_box_var_data) {
      LOG(INFO) << "prior_box_var_data not null" << std::endl;
      if (axis == 0) {
        LOG(INFO) << "use DecodeCenterSize<1, 2> axis == 0" << std::endl;
        DecodeCenterSize<0, 2>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      } else {
        LOG(INFO) << "use DecodeCenterSize<1, 2> axis == 1" << std::endl;
        DecodeCenterSize<1, 2>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      }
    } else if (!(variance.empty())) {
      LOG(INFO) << "prior_box_var_data null" << std::endl;
      if (axis == 0) {
        DecodeCenterSize<0, 1>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      } else {
        DecodeCenterSize<1, 1>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      }
    } else {
      if (axis == 0) {
        DecodeCenterSize<0, 0>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      } else {
        DecodeCenterSize<1, 0>(target_box_data,
                               prior_box_data,
                               prior_box_var_data,
                               target_box_shape,
                               prior_box_shape,
                               prior_box_var_shape,
                               normalized,
                               variance,
                               output_data);
      }
    }
  }
}

void box_coder_ref(const std::shared_ptr<operators::BoxCoderOpLite> op) {
  Scope *scope = op->scope();
  const OpInfo *op_info = op->op_info();
  auto prior_box =
      scope->FindVar(op_info->Input("PriorBox").front())->GetMutable<Tensor>();
  auto target_box =
      scope->FindVar(op_info->Input("TargetBox").front())->GetMutable<Tensor>();
  auto box_var = scope->FindVar(op_info->Input("PriorBoxVar").front())
                     ->GetMutable<Tensor>();
  auto output_box = scope->FindVar(op_info->Output("OutputBox").front())
                        ->GetMutable<Tensor>();

  auto code_type_str = op_info->GetAttr<std::string>("code_type");
  auto box_normalized = op_info->GetAttr<bool>("box_normalized");
  auto axis = op_info->GetAttr<int>("axis");
  auto code_type = GetBoxCodeType(code_type_str);
  std::vector<float> variance;
  if (op_info->HasAttr("variance")) {
    variance = op_info->GetAttr<std::vector<float>>("variance");
  }
  Compute(code_type,
          prior_box,
          target_box,
          box_var,
          output_box,
          variance,
          box_normalized,
          axis);
}

void test_box_coder(int row,
                    int col,
                    int len,
                    int axis,
                    cnmlBoxCodeType_t code_type,
                    bool box_normalized) {
  // prepare input&output variables
  Scope scope;
  std::string prior_box_var_name("PriorBox");
  std::string taget_box_var_name("TargetBox");
  std::string output_box_var_name("OutputBox");
  std::string box_var_var_name("PriorBoxVar");
  std::string output_ref_var_name("OutputBox_ref");
  auto *prior_box = scope.Var(prior_box_var_name)->GetMutable<Tensor>();
  auto *target_box = scope.Var(taget_box_var_name)->GetMutable<Tensor>();
  auto *box_var = scope.Var(box_var_var_name)->GetMutable<Tensor>();
  auto *output_box = scope.Var(output_box_var_name)->GetMutable<Tensor>();
  auto *output_box_ref = scope.Var(output_ref_var_name)->GetMutable<Tensor>();

  if (code_type == cnmlBoxCodeType_t::Encode) {
    // target_box_shape = {row, len};
    // prior_box_shape = {col, len};
    // output_shape = {row, col, len};
    target_box->Resize({row, len});
    prior_box->Resize({col, len});
    box_var->Resize({col, len});
  } else if (code_type == cnmlBoxCodeType_t::Decode) {
    // target_box_shape = {row,col,len};
    // prior_box_shape = {col, len} if axis == 0, or {row, len};
    // output_shape = {row, col, len};
    target_box->Resize({row, col, len});
    if (axis == 0) {
      prior_box->Resize({col, len});
      box_var->Resize({col, len});
    } else if (axis == 1) {
      prior_box->Resize({row, len});
      box_var->Resize({row, len});
    } else {
      LOG(FATAL) << "axis should in {0,1} ,but got " << axis << std::endl;
    }
  }

  // initialize input&output data
  // FillTensor<float>(prior_box);
  // FillTensor<float>(target_box);
  // FillTensor<float, int>(box_var); // ??????
  for (int i = 0; i < prior_box->dims().production(); i++) {
    prior_box->mutable_data<float>()[i] = static_cast<float>((i % 8) + 1);
  }
  for (int i = 0; i < target_box->dims().production(); i++) {
    target_box->mutable_data<float>()[i] = static_cast<float>((i % 8) + 1);
  }
  for (int i = 0; i < box_var->dims().production() / 4; i++) {
    box_var->mutable_data<float>()[i * 4 + 0] = 0.1;
    box_var->mutable_data<float>()[i * 4 + 1] = 0.1;
    box_var->mutable_data<float>()[i * 4 + 2] = 0.2;
    box_var->mutable_data<float>()[i * 4 + 3] = 0.2;
  }

  LOG(INFO) << "prior_box count : " << prior_box->dims().production();
  LOG(INFO) << "target_box count : " << target_box->dims().production();
  LOG(INFO) << "box_var count : " << box_var->dims().production();

  // ToFile(*prior_box, "prior_box.txt");
  // ToFile(*box_var, "box_var.txt");
  // ToFile(*target_box, "target_box.txt");

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("box_coder");
  opdesc.SetInput("PriorBox", {prior_box_var_name});
  opdesc.SetInput("TargetBox", {taget_box_var_name});
  opdesc.SetInput("PriorBoxVar", {box_var_var_name});
  opdesc.SetOutput("OutputBox", {output_box_var_name});

  opdesc.SetAttr("axis", axis);
  opdesc.SetAttr("box_normalized", box_normalized);
  opdesc.SetAttr("code_type", BoxCodeTypeToStr(code_type));

  // trans inputs
  Tensor prior_box_trans;
  Tensor box_var_trans;
  Tensor target_box_trans;
  prior_box_trans.Resize(prior_box->dims());
  box_var_trans.Resize(box_var->dims());
  target_box_trans.Resize(target_box->dims());

  auto op = CreateOp<paddle::lite::operators::BoxCoderOpLite>(opdesc, &scope);
  box_coder_ref(op);
  output_box_ref->CopyDataFrom(*output_box);

  // transpose(prior_box->mutable_data<float>(),
  //           prior_box_trans.mutable_data<float>(),
  //           {static_cast<int>(prior_box->dims()[0]),
  //            static_cast<int>(prior_box->dims()[1]),
  //            1,
  //            1},
  //           {0, 2, 3, 1});

  // row col len 1 --> row len 1 col
  transpose(target_box->mutable_data<float>(),
            target_box_trans.mutable_data<float>(),
            {
                static_cast<int>(target_box->dims()[0]),
                static_cast<int>(target_box->dims()[1]),
                static_cast<int>(target_box->dims()[2]),
                1,
            },
            {0, 2, 3, 1});

  // transpose(box_var->mutable_data<float>(),
  //           box_var_trans.mutable_data<float>(),
  //           {static_cast<int>(box_var->dims()[0]),
  //            static_cast<int>(box_var->dims()[0]),
  //            1,
  //            1},
  //           {0, 2, 3, 1});

  target_box->CopyDataFrom(target_box_trans);

  LaunchOp(op,
           {prior_box_var_name, taget_box_var_name, box_var_var_name},
           {output_box_var_name});

  // execute reference implementation and save to output tensor('out')

  // compare results
  auto *output_data = output_box->mutable_data<float>();
  auto *output_ref_data = output_box_ref->mutable_data<float>();
  Tensor output_trans;
  output_trans.Resize(output_box->dims());
  // row * len * 1 * col -> row * col * len * 1
  transpose(output_data,
            output_trans.mutable_data<float>(),
            {static_cast<int>(output_box->dims()[0]),
             static_cast<int>(output_box->dims()[2]),
             1,
             static_cast<int>(output_box->dims()[1])},
            {0, 3, 1, 2});

  output_data = output_trans.mutable_data<float>();
  // ToFile(*output_box, "output_mlu_before_trans.txt");
  // ToFile(&output_trans, "output_mlu.txt");
  // ToFile(output_box_ref, "output_cpu.txt");
  for (int i = 0; i < output_box->dims().production(); i++) {
    VLOG(6) << i;
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, prior_density_box) {
  int row = 1;
  int col = 20560;
  int len = 4;
  int axis = 0;
  cnmlBoxCodeType_t code_type = cnmlBoxCodeType_t::Decode;
  bool box_normalized = true;
  test_box_coder(row, col, len, axis, code_type, box_normalized);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(box_coder, kMLU);
