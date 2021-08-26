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

#include "lite/operators/conv_op.h"

#include <algorithm>

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto* graph = static_cast<Graph*>(ctx);
  const auto* op_info = op->op_info();
  const auto* scope = op->scope();
  VLOG(3) << "[MLU] Converting " << op_info->Type() << "... ";
  CHECK(!op_info->HasAttr("act_type"));

  // get input, filter and op attributes
  const auto input_var_name = op_info->Input("Input").front();
  const auto& input_dims =
      scope->FindVar(input_var_name)->GetMutable<Tensor>()->dims();
  const auto filter_var_name = op_info->Input("Filter").front();
  auto* filter = scope->FindVar(filter_var_name)->GetMutable<Tensor>();
  const auto& filter_dims = filter->dims();
  const auto output_var_name = op_info->Output("Output").front();
  auto* output = scope->FindVar(output_var_name)->GetMutable<Tensor>();
  const auto output_shape = output->dims().Vectorize();
  const auto bs = input_dims[0];
  const auto oc = filter_dims[0];
  const auto groups = op_info->GetAttr<int>("groups");

  CHECK_EQ(input_dims.size(), 4u);
  CHECK_EQ(filter_dims.size(), 4u);
  CHECK(!(op_info->HasAttr("fuse_relu") &&
          (op_info->GetAttr<bool>("fuse_relu") == true)))
      << "UnSupported param fuse_relu is true!";
  const auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  CHECK_EQ(strides.size(), 2u);
  CHECK_EQ(dilations.size(), 2u);
  if (paddings.size() == 2u) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4u)
      << "Paddings size should be the same or twice as the input size.";

  const std::string padding_algorithm =
      op_info->HasAttr("padding_algorithm")
          ? op_info->GetAttr<std::string>("padding_algorithm")
          : "";

  operators::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      strides,
                                      padding_algorithm,
                                      input_dims,
                                      filter_dims);
  bool is_group_mode = groups > 1;

  bool is_depthwise_mode = false;
  if (filter_dims[0] == groups && filter_dims[1] == 1 && dilations[0] == 1 &&
      dilations[1] == 1) {  // depthwise filter shape = {1, ic ,kh ,kw}
    is_depthwise_mode = true;
    is_group_mode = false;
  }

  auto input_tensor = graph->GetNode(input_var_name);
  const auto output_tensor = graph->AddNode(
      output_var_name, output_shape, CNML_TENSOR, CNML_NCHW, graph->FPType());
  std::vector<int64_t> cnml_filter_shape = {
      filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]};
  if (is_depthwise_mode) {
    /*paddle filter shape is {oc , ic / groups == 1, kh, kw} while
     cnml depthwise conv filter expect shape {oc / groups == 1 , ic , kh, kw}
     so we should shape filter shape
     */
    cnml_filter_shape = {
        filter_dims[1], filter_dims[0], filter_dims[2], filter_dims[3]};
  }

  // Create filter node
  const auto filter_tensor = graph->AddNode(filter_var_name,
                                            cnml_filter_shape,
                                            CNML_FILTER,
                                            CNML_NCHW,
                                            graph->FPType());
  const auto weight_scale = op_info->GetInputScale(filter_var_name);

  if (filter->precision() == PrecisionType::kUnk ||
      filter->precision() == PrecisionType::kInt8) {
    std::vector<float> filter_dequant(filter->data_size());
    dequant(filter_dequant.data(),
            filter->mutable_data<int8_t>(),
            1,
            cnml_filter_shape[0],
            cnml_filter_shape[1] * cnml_filter_shape[2] * cnml_filter_shape[3],
            weight_scale);
    transpose(filter_dequant.data(),
              filter->mutable_data<float>(),
              {static_cast<int>(cnml_filter_shape[0]),
               static_cast<int>(cnml_filter_shape[1]),
               static_cast<int>(cnml_filter_shape[2]),
               static_cast<int>(cnml_filter_shape[3])},
              {0, 2, 3, 1});
    filter->set_precision(PrecisionType::kFloat);
  } else if (filter->precision() != PrecisionType::kFloat) {
    LOG(FATAL) << "UnSupported weight precision!";
  }

  std::string bias_var_name;
  std::shared_ptr<MLUTensor> bias_tensor;
  if (HasInputArg(op_info, scope, "Bias")) {
    const DDim output_dims(output_shape);
    bias_var_name = op_info->Input("Bias").front();
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
    const auto& bias_dims = bias->dims();
    const auto bias_data_size = bias_dims.production();
    const auto output_data_size = output_dims.production();
    std::vector<int64_t> bias_shape;
    if (bias_data_size == oc) {
      // 0: {oc}
      bias_shape = {1, 1, 1, oc};
    } else if (bias_data_size == output_data_size / bs) {
      LOG(FATAL) << "Unsupported ... ...";
      // 1: {1, oc, oh, ow}
      bias_shape = {1, output_dims[1], output_dims[2], output_dims[3]};
    } else if (bias_data_size == output_data_size) {
      LOG(FATAL) << "Unsupported ... ...";
      // 2: {n, oc, oh, ow}
      bias_shape = output_dims.Vectorize();
    } else {
      LOG(ERROR) << "[MLU] Bias dimension " << bias_dims
                 << " isn't supported in conv2d Op when output dimension is "
                 << output_dims;
    }
    bias_tensor = graph->AddNode(
        bias_var_name, bias_shape, CNML_CONST, CNML_NHWC, graph->FPType());
    graph->BindConstData(bias_var_name, bias);
  }

  const auto input_scale = op_info->GetInputScale(input_var_name)[0];

  bool use_first_conv = false;
  if (lite::TargetWrapperMlu::UseFirstConv() && input_dims[1] == 3) {
    use_first_conv = true;
  }

  cnmlBaseOp_t conv_op;
  if (use_first_conv) {
    cnmlConvFirstOpParam_t conv_param;
    CNML_CALL(cnmlCreateConvFirstOpParam_V2(&conv_param,
                                            strides[0],
                                            strides[1],
                                            dilations[0],
                                            dilations[1],
                                            paddings[2],
                                            paddings[2],
                                            paddings[0],
                                            paddings[0]));
    const auto mean_tensor = graph->AddNode("first_conv_mean_tensor",
                                            std::vector<int64_t>{1, 1, 1, 3},
                                            CNML_CONST,
                                            CNML_NHWC,
                                            graph->FPType());
    const auto std_tensor = graph->AddNode("first_conv_std_tensor",
                                           std::vector<int64_t>{1, 1, 1, 3},
                                           CNML_CONST,
                                           CNML_NHWC,
                                           graph->FPType());

    graph->BindConstRawData("first_conv_mean_tensor",
                            lite::TargetWrapperMlu::MeanVec().data(),
                            3,
                            false);
    graph->BindConstRawData("first_conv_std_tensor",
                            lite::TargetWrapperMlu::StdVec().data(),
                            3,
                            false);

    input_tensor->set_mlu_dtype(CNML_DATA_UINT8);
    CNML_CALL(cnmlCreateConvFirstOpForward(
        &conv_op,
        conv_param,
        input_tensor->mlu_tensor(),
        mean_tensor->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr,
        std_tensor->mlu_tensor()));
    CNML_CALL(cnmlDestroyConvFirstOpParam(&conv_param));
  } else if (is_depthwise_mode) {
    cnmlConvDepthwiseOpParam_t conv_depthwise_param;
    cnmlCreateConvDepthwiseOpParam_V2(&conv_depthwise_param,
                                      strides[0],
                                      strides[1],
                                      paddings[0] * 2,
                                      paddings[2] * 2);
    CNML_CALL(cnmlCreateConvDepthwiseOpForward(
        &conv_op,
        conv_depthwise_param,
        input_tensor->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr));
    CNML_CALL(cnmlDestroyConvDepthwiseOpParam(&conv_depthwise_param));
  } else if (is_group_mode) {
    cnmlConvOpParam_t conv_param;
    CNML_CALL(cnmlCreateConvOpParam(&conv_param,
                                    strides[0],
                                    strides[1],
                                    dilations[0],
                                    dilations[1],
                                    paddings[0] * 2,
                                    paddings[2] * 2));
    CNML_CALL(cnmlCreateConvGroupOpForward(
        &conv_op,
        conv_param,
        input_tensor->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr,
        groups));
    CNML_CALL(cnmlDestroyConvOpParam(&conv_param));
  } else {
    cnmlConvOpParam_t conv_param;
    VLOG(5) << "conv param (" << input_var_name << ")"
            << "stride: " << strides[0] << ',' << strides[1] << '\t'
            << "dilations: " << dilations[0] << ',' << dilations[1] << '\t'
            << "paddings: " << paddings[0] << ',' << paddings[2] << std::endl;
    CNML_CALL(cnmlCreateConvOpParam(&conv_param,
                                    strides[0],
                                    strides[1],
                                    dilations[0],
                                    dilations[1],
                                    paddings[0] * 2,
                                    paddings[2] * 2));
    CNML_CALL(cnmlCreateConvOpForward(
        &conv_op,
        conv_param,
        input_tensor->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr));
    CNML_CALL(cnmlDestroyConvOpParam(&conv_param));
  }

  if (!is_depthwise_mode) {
    graph->SetComputingDataType(
        conv_op, graph->GetNode(input_var_name)->mlu_tensor(), 1 / input_scale);
    graph->SetComputingDataType(
        conv_op,
        filter_tensor->mlu_tensor(),
        1 / *max_element(weight_scale.begin(), weight_scale.end()));
  }
  CNML_CALL(cnmlSetOperationComputingLayout(conv_op, CNML_NHWC));
  if (HasInputArg(op_info, scope, "Bias")) {
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
    graph->BindConstData(bias_var_name, bias);
  }
  graph->BindConstData(filter_var_name, filter);
  graph->FuseOp(conv_op);
  CNML_CALL(cnmlDestroyBaseOp(&conv_op));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kMLU,
                         paddle::lite::subgraph::mlu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kMLU,
                         paddle::lite::subgraph::mlu::ConvConverter);
