// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int PadConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("X").front();
  auto input_scale_name = "X0_scale";
  auto has_input_scale = op_info->HasInputScale(input_scale_name, true);
  auto input_scale =
      has_input_scale ? op_info->GetInputScale(input_scale_name, true)[0] : 0.f;
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto input_rank = input_dims.size();
  auto output_name = op_info->Output("Out").front();
  auto output_scale_name = "Out0_scale";
  auto has_output_scale = op_info->HasOutputScale(output_scale_name, true);
  auto output_scale = has_output_scale
                          ? op_info->GetOutputScale(output_scale_name, true)[0]
                          : 0.f;
  auto output = scope->FindMutableTensor(output_name);
  auto mode = op_info->GetAttr<std::string>("mode");
  float value;
  std::vector<int> pads;
  if (op_type == "pad2d") {
    value = op_info->GetAttr<float>("pad_value");
    if (op_info->HasAttr("variable_padding") &&
        op_info->GetAttr<bool>("variable_paddings")) {
      auto Paddings =
          scope->FindMutableTensor(op_info->Input("Paddings").front());
      auto ptr = Paddings->data<int>();
      if (Paddings->dims().size() < 4) {
        LOG(FATAL) << "Paddings size must be four: %d \n",
            static_cast<int>(Paddings->dims().size());
        return FAILED;
      }
      pads = {ptr[0], ptr[1], ptr[2], ptr[3]};
    } else {
      pads = op_info->GetAttr<std::vector<int>>("paddings");
    }
  } else {
    value = op_info->GetAttr<float>("pad_value");
    pads = op_info->GetAttr<std::vector<int>>("paddings");
  }

  auto data_format = op_info->GetAttr<std::string>("data_format");
  std::vector<int> paddings;
  if (data_format == "NCDHW") {
    CHECK_EQ(pads.size(), 6);
    paddings = {
        0, 0, 0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1]};
    output->Resize({input_dims[0],
                    input_dims[1],
                    input_dims[2] + pads[4] + pads[5],
                    input_dims[3] + pads[2] + pads[3],
                    input_dims[4] + pads[0] + pads[1]});
  } else if (data_format == "NDHWC") {
    CHECK_EQ(pads.size(), 6);
    paddings = {
        0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1], 0, 0};
    output->Resize({input_dims[0],
                    input_dims[1] + pads[4] + pads[5],
                    input_dims[2] + pads[2] + pads[3],
                    input_dims[3] + pads[0] + pads[1],
                    input_dims[4]});
  } else if (data_format == "NCHW") {
    CHECK_EQ(pads.size(), 4);
    paddings = {0, 0, 0, 0, pads[0], pads[1], pads[2], pads[3]};
    output->Resize({input_dims[0],
                    input_dims[1],
                    input_dims[2] + pads[0] + pads[1],
                    input_dims[3] + pads[2] + pads[3]});
  } else if (data_format == "NHWC") {
    CHECK_EQ(pads.size(), 4);
    paddings = {0, 0, pads[0], pads[1], pads[2], pads[3], 0, 0};
    output->Resize({input_dims[0],
                    input_dims[1] + pads[0] + pads[1],
                    input_dims[2] + pads[2] + pads[3],
                    input_dims[3]});
  } else if (data_format == "NCL") {
    CHECK_EQ(pads.size(), 2);
    paddings = {0, 0, 0, 0, pads[0], pads[1]};
    output->Resize(
        {input_dims[0], input_dims[1], input_dims[2] + pads[0] + pads[1]});
  } else if (data_format == "NLC") {
    CHECK_EQ(pads.size(), 2);
    paddings = {0, 0, pads[0], pads[1], 0, 0};
    output->Resize(
        {input_dims[0], input_dims[1] + pads[0] + pads[1], input_dims[2]});
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format;
  }
  CHECK_EQ(paddings.size(), 2 * input_rank);
  auto output_dims = output->dims();

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(input_name)) {
    input_operand = converter->GetOperand(input_name);
  } else {
    if (has_input_scale) {
      input_operand = converter->AddQuant8VariableOperand(
          input_dims, input_scale, input_name);
    } else {
      input_operand =
          converter->AddFloat32VariableOperand(input_dims, input_name);
    }
  }

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_output_scale) {
    output_operand = converter->AddQuant8VariableOperand(
        output_dims, output_scale, output_name);
  } else {
    output_operand =
        converter->AddFloat32VariableOperand(output_dims, output_name);
  }

  // Pads, mode, value
  int mode_code = PadMode2NNAdapterPadModeCode(mode);
  auto pads_operand = converter->AddInt32ConstantOperand(
      &paddings[0], DDim({static_cast<int64_t>(paddings.size())}));
  auto mode_operand = converter->AddInt32ConstantOperand(mode_code);
  auto value_operand = converter->AddFloat32ConstantOperand(value);

  // Pad operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, pads_operand, mode_operand, value_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_PAD, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pad3d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::PadConverter);
REGISTER_SUBGRAPH_BRIDGE(pad2d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::PadConverter);
