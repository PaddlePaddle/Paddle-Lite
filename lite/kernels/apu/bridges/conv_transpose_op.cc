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

#include <vector>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/apu/bridges/utility.h"

#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int ConvTransposeConverter(void *ctx, OpLite *op, KernelBase *kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph *>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  int neuron_errCode;
  VLOG(3) << "[APU] Converting [" << op_type << "]";

  CHECK(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"));

  // Get input, output and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_EQ(input_dims.size(), 4);

  auto filter_name = op_info->Input("Filter").front();
  auto filter_scale_name = "Filter0_scale";
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();
  CHECK_EQ(filter_dims.size(), 4);

  auto output_name = op_info->Output("Output").front();
  auto output_scale_name = "Output0_scale";

  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  CHECK_EQ(strides.size(), 2L);
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  if (groups > 1) {
    LOG(WARNING) << "[NPU] only support groups == 1";
    return FAILED;
  }

  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;
  auto fuse_relu =
      op_info->HasAttr("fuse_relu") && op_info->GetAttr<bool>("fuse_relu");

  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  CHECK_EQ(dilations.size(), 2L);
  std::string padding_algorithm =
      op_info->HasAttr("padding_algorithm")
          ? op_info->GetAttr<std::string>("padding_algorithm")
          : "";
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }

  CHECK_EQ(paddings.size(), 4L)
      << "[APU] Paddings size should be the same or twice as the input size."
      << paddings.size();

  operators::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      strides,
                                      padding_algorithm,
                                      input_dims,
                                      filter_dims);

  std::vector<int> output_dims;
  // Set output_dims: batches
  output_dims.push_back(input_dims[0]);

  std::vector<int> output_size;
  if (op_info->HasAttr("output_size")) {
    output_size = op_info->GetAttr<std::vector<int>>("output_size");
  }

  if (output_size.size() > 2) {
    // Set output_dims: height, width
    output_dims.push_back(output_size[0]);
    output_dims.push_back(output_size[1]);
  } else {
    // Compute output size
    for (int i = 0; i < strides.size(); i++) {
      int kernel_ext = filter_dims[i + 2];
      int output_size = (input_dims[i + 2] - 1) * strides[i] + kernel_ext -
                        paddings[i * 2] - paddings[i * 2 + 1];
      output_dims.push_back(output_size);
    }
  }
  output_dims.push_back(filter_dims[1]);

  CHECK(op_info->HasInputScale(input_scale_name, true));
  auto input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  CHECK(op_info->HasInputScale(filter_scale_name, true));
  auto filter_scale = op_info->GetInputScale(filter_scale_name, true);
  CHECK(op_info->HasOutputScale(output_scale_name, true));
  auto output_scale = op_info->GetOutputScale(output_scale_name, true)[0];

  VLOG(3) << "strides.size(): " << strides.size() << " ,groups: " << groups
          << " ,dilations: " << dilations[0] << ":" << dilations[1];
  VLOG(3) << "with_act: " << with_act << " ,act_type: " << act_type;
  VLOG(3) << "input_dims: " << input_dims
          << " ,filter_scale size: " << filter_scale.size();
  VLOG(3) << "filter_dims(Cin, Cout, H, W): " << filter_dims
          << " ,memory_size: " << filter->memory_size()
          << " ,data_size: " << filter->data_size();

  // Add input tensor type
  NeuronOperandType inType;
  inType.type = NEURON_TENSOR_QUANT8_ASYMM;
  inType.scale = input_scale;
  inType.zeroPoint = 128;
  inType.dimensionCount = input_dims.size();
  std::vector<uint32_t> dims_in = {(uint32_t)input_dims[0],
                                   (uint32_t)input_dims[2],
                                   (uint32_t)input_dims[3],
                                   (uint32_t)input_dims[1]};
  inType.dimensions = &dims_in[0];

  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    VLOG(3) << "Graph has " << input_name;
    // Input operand already created by previous OP
    input_node = graph->Get(input_name);
  } else {
    // Add input operand
    if (graph->IsInput(input_name)) {
      // Insert transpose for NCHW -> NHWC
      insert_transpose_node(ctx,
                            input_name,
                            "transpose_" + input_name,
                            {(uint32_t)input_dims[0],
                             (uint32_t)input_dims[1],
                             (uint32_t)input_dims[2],
                             (uint32_t)input_dims[3]},
                            dims_in,
                            {0, 2, 3, 1},
                            inType.scale,
                            inType.zeroPoint);

      // Change input_name because we add transpose op
      input_name = "transpose_" + input_name;
      input_node = graph->Get(input_name);
      if (input_node == nullptr) return subgraph::FAILED;
    } else {
      NeuronModel_addOperand(model, &inType);
      input_node = graph->Add(input_name, dims_in);
    }
  }

  VLOG(3) << "input node idx: " << input_node->index()
          << ": input_scale: " << input_scale
          << ", inType: " << inType.dimensions[0] << ":" << inType.dimensions[1]
          << ":" << inType.dimensions[2] << ":" << inType.dimensions[3];

  // Add bias type
  NeuronOperandType biasType;

  // Add filter type
  // Relay out filter (Cin,Cout,H,W) -> (depth_out, h, w, depth_in)
  Tensor transpose_filter;
  std::vector<uint32_t> dims_filter;
  transpose_filter.Resize({(uint32_t)filter_dims[1],
                           (uint32_t)filter_dims[2],
                           (uint32_t)filter_dims[3],
                           (uint32_t)filter_dims[0]});

  transposeAsym(filter->data<int8_t>(),
                transpose_filter.mutable_data<uint8_t>(),
                {(uint32_t)filter_dims[0],
                 (uint32_t)filter_dims[1],
                 (uint32_t)filter_dims[2],
                 (uint32_t)filter_dims[3]},
                {1, 2, 3, 0});

  dims_filter = {(uint32_t)filter_dims[1],
                 (uint32_t)filter_dims[2],
                 (uint32_t)filter_dims[3],
                 (uint32_t)filter_dims[0]};

  NeuronOperandType filterType;
  filterType.type = NEURON_TENSOR_QUANT8_ASYMM;
  filterType.scale = filter_scale[0];
  filterType.zeroPoint = 128;
  filterType.dimensionCount = filter_dims.size();
  filterType.dimensions = &dims_filter[0];
  biasType.scale = inType.scale * filterType.scale;

  std::shared_ptr<Node> filter_node = nullptr;
  NeuronModel_addOperand(model, &filterType);
  filter_node = graph->Add(filter_name, dims_filter);
  auto precision = filter->precision();
  VLOG(3) << " filter node idx: " << filter_node->index()
          << " filter_scale[0]=" << filter_scale[0]
          << " filter memory_size=" << filter->memory_size()
          << " filter precision=" << PrecisionToStr(precision)
          << " :filterType: " << filterType.dimensions[0] << ":"
          << filterType.dimensions[2] << ":" << filterType.dimensions[2] << ":"
          << filterType.dimensions[3];

  memcpy(filter->mutable_data<int8_t>(),
         transpose_filter.mutable_data<uint8_t>(),
         filter->memory_size());

  // Set filter value
  neuron_errCode = NeuronModel_setOperandValue(
      model, filter_node->index(), filter->raw_data(), filter->memory_size());
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Set filter operand value fail:" << neuron_errCode;
    return subgraph::FAILED;
  }

  // Add biasType node value
  // A 1-D tensor, of shape [depth_out], specifying the bias.
  // For filter tensor of NEURON_TENSOR_QUANT8_ASYMM, the bias should be of
  // NEURON_TENSOR_INT32 with zeroPoint of 0 and bias_scale ==
  // input_scale * filter_scale
  biasType.type = NEURON_TENSOR_INT32;
  biasType.zeroPoint = 0;
  std::vector<uint32_t> dims_bias;
  std::shared_ptr<Node> bias_node = nullptr;

  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    auto channel_size = bias->dims().production();
    CHECK_EQ(channel_size, filter_dims[1] * groups);
    CHECK_EQ(bias_dims.size(), 1);

    biasType.dimensionCount = bias_dims.size();
    for (int i = 0; i < bias_dims.size(); i++)
      dims_bias.push_back(bias_dims[i]);
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType);  // Operand 2: bias
    bias_node = graph->Add(bias_name, dims_bias);
    VLOG(3) << "node idx: " << bias_node->index()
            << ": Bias name: " << bias_name
            << " ,bias scale: " << biasType.scale
            << " ,dimensions: " << bias_dims
            << " ,channel_size:" << channel_size;

  } else {
    // Create default bias with value 0
    biasType.dimensionCount = 1;
    dims_bias = {(uint32_t)output_dims[1]};
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType);  // Operand 2: bias
    bias_node = graph->Add(filter_name + "_default_bias", dims_bias);
    VLOG(3) << "node idx: " << bias_node->index()
            << ": Bias name: default_bias "
            << " ,bias scale: " << biasType.scale
            << " ,dimensions: " << dims_bias.size();
  }

  NeuronOperandType int32Type;
  int32Type.type = NEURON_INT32;
  int32Type.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {1};

  std::shared_ptr<Node> paddingL_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 3: padding left
  paddingL_node = graph->Add(filter_name + "_padding_left", dims_int32);

  std::shared_ptr<Node> paddingR_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 4: padding right
  paddingR_node = graph->Add(filter_name + "_padding_right", dims_int32);

  std::shared_ptr<Node> paddingT_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 5: padding top
  paddingT_node = graph->Add(filter_name + "_padding_top", dims_int32);

  std::shared_ptr<Node> paddingB_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 6: padding bottom
  paddingB_node = graph->Add(filter_name + "_padding_bottom", dims_int32);

  std::shared_ptr<Node> strideW_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 7: stride width
  strideW_node = graph->Add(filter_name + "_stride_width", dims_int32);

  std::shared_ptr<Node> strideH_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 8: stride height
  strideH_node = graph->Add(filter_name + "_stride_height", dims_int32);

  std::shared_ptr<Node> fuse_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 9: fuse
  fuse_node = graph->Add(filter_name + "_fuse", dims_int32);

  NeuronOperandType boolType;
  boolType.type = NEURON_BOOL;
  boolType.dimensionCount = 0;  // Must be 0 for scalars.
  std::shared_ptr<Node> layout_node = nullptr;
  NeuronModel_addOperand(model, &boolType);  // Operand 9: fuse
  layout_node = graph->Add(filter_name + "_layout", dims_int32);

  // Add output tensor type
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = output_scale;
  outType.zeroPoint = 128;
  outType.dimensionCount = output_dims.size();
  std::vector<uint32_t> dims_out = {(uint32_t)output_dims[0],
                                    (uint32_t)output_dims[1],
                                    (uint32_t)output_dims[2],
                                    (uint32_t)output_dims[3]};
  outType.dimensions = &dims_out[0];
  std::shared_ptr<Node> output_node = nullptr;
  if (graph->Has(output_name)) {
    output_node = graph->Get(output_name);
  } else {
    if (graph->IsOutput(output_name)) {
      NeuronModel_addOperand(model, &outType);
      output_node = graph->Add("transpose_" + output_name, dims_out);
    } else {
      NeuronModel_addOperand(model, &outType);
      output_node = graph->Add(output_name, dims_out);
    }
  }
  VLOG(3) << "output node idx: " << output_node->index()
          << ": output_scale: " << outType.scale
          << " ,outType: " << outType.dimensions[0] << ":"
          << outType.dimensions[1] << ":" << outType.dimensions[2] << ":"
          << outType.dimensions[3];

  // Add bias value
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);

    int32_t *int32_bias_data =
        reinterpret_cast<int32_t *>(bias->mutable_data<float>());
    float2int32(
        bias->data<float>(), input_scale, filter_scale, int32_bias_data);

    VLOG(3) << "int32_bias_data: " << int32_bias_data[0] << ":"
            << int32_bias_data[1] << ":" << int32_bias_data[2] << ":"
            << int32_bias_data[3];

    neuron_errCode = NeuronModel_setOperandValue(
        model, bias_node->index(), bias->raw_data(), bias->memory_size());
  } else {
    auto int32_bias = std::make_shared<Tensor>();
    int32_bias->Resize({1, output_dims[3]});
    int32_bias->mutable_data<int32_t>();
    VLOG(3) << "bais_default: " << int32_bias->memory_size();
    memset(int32_bias->mutable_data<int32_t>(), 0, int32_bias->memory_size());
    neuron_errCode = NeuronModel_setOperandValue(model,
                                                 bias_node->index(),
                                                 int32_bias->raw_data(),
                                                 int32_bias->memory_size());
    bias_node->set_data(int32_bias);
  }
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Set bias operand value fail:" << neuron_errCode;
    return subgraph::FAILED;
  }

  VLOG(3) << "paddings: " << paddings[0] << ":" << paddings[1] << ":"
          << paddings[2] << ":" << paddings[3];
  // Add padding value
  int32_t padding_val[1];
  padding_val[0] = paddings[2];
  NeuronModel_setOperandValue(
      model, paddingL_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[3];
  NeuronModel_setOperandValue(
      model, paddingR_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[0];
  NeuronModel_setOperandValue(
      model, paddingT_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[1];
  NeuronModel_setOperandValue(
      model, paddingB_node->index(), padding_val, sizeof(int32_t) * 1);

  VLOG(3) << " stride width:" << strides[1] << " height:" << strides[0];

  // Add Stride
  int32_t stride_val[1];
  stride_val[0] = strides[1];  // entry 1: width stride
  NeuronModel_setOperandValue(
      model, strideW_node->index(), stride_val, sizeof(int32_t) * 1);
  stride_val[0] = strides[0];  // entry 0: height stride
  NeuronModel_setOperandValue(
      model, strideH_node->index(), stride_val, sizeof(int32_t) * 1);

  int32_t fuse_val[1] = {NEURON_FUSED_NONE};
  if (act_type == "relu") {
    fuse_val[0] = NEURON_FUSED_RELU;
  } else if (act_type == "relu1") {
    fuse_val[0] = NEURON_FUSED_RELU1;
  } else if (act_type == "relu6") {
    fuse_val[0] = NEURON_FUSED_RELU6;
  } else if (!act_type.empty()) {
    fuse_val[0] = NEURON_FUSED_NONE;
    LOG(WARNING) << "Support act_type: " << act_type;
    return FAILED;
  }

  NeuronModel_setOperandValue(
      model, fuse_node->index(), fuse_val, sizeof(int32_t) * 1);

  bool layout_val[] = {false};
  NeuronModel_setOperandValue(
      model, layout_node->index(), layout_val, sizeof(bool) * 1);

  std::vector<uint32_t> addInIndex = {
      input_node->index(),     // 0: input
      filter_node->index(),    // 1: filter
      bias_node->index(),      // 2: bias
      paddingL_node->index(),  // 3: padding left
      paddingR_node->index(),  // 4: padding right
      paddingT_node->index(),  // 5: padding top
      paddingB_node->index(),  // 6: padding bottom
      strideW_node->index(),   // 7: stride width
      strideH_node->index(),   // 8: stride height
      fuse_node->index(),      // 9: fuse
      layout_node->index()};   // 10: layout

  std::vector<uint32_t> addOutIndex = {output_node->index()};
  neuron_errCode = NeuronModel_addOperation(model,
                                            NEURON_TRANSPOSE_CONV_2D,
                                            addInIndex.size(),
                                            &addInIndex[0],
                                            addOutIndex.size(),
                                            &addOutIndex[0]);

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Add op fail:" << op_type;
    return FAILED;
  }

  if (graph->IsOutput(output_name)) {
    // Insert transpose for NHWC -> NCHW
    insert_transpose_node(ctx,
                          "transpose_" + output_name,
                          output_name,
                          dims_out,
                          {(uint32_t)output_dims[0],
                           (uint32_t)output_dims[1],
                           (uint32_t)output_dims[2],
                           (uint32_t)output_dims[3]},
                          {0, 3, 1, 2},
                          outType.scale,
                          outType.zeroPoint);
    output_node = graph->Get(output_name);
    if (output_node == nullptr) return subgraph::FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d_transpose,
                         kAPU,
                         paddle::lite::subgraph::apu::ConvTransposeConverter);
