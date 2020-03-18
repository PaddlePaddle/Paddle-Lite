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
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

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

  // Get input, filter and op attributes
  const auto input_var_name = op_info->Input("Input").front();
  const auto& input_dims_nhwc =
      scope->FindVar(input_var_name)->GetMutable<Tensor>()->dims();
  const auto input_dims = DimNHWC2NCHW(input_dims_nhwc);
  const auto filter_var_name = op_info->Input("Filter").front();
  auto* filter = scope->FindVar(filter_var_name)->GetMutable<Tensor>();
  const auto& filter_dims = filter->dims();
  const auto output_var_name = op_info->Output("Output").front();
  const auto bs = input_dims[0];
  const auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  const auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
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

  std::vector<int64_t> output_shape({bs, oc});
  for (size_t i = 0; i < 2; i++) {
    const int dkernel = dilations[i] * (filter_dims[2 + i] - 1) + 1;
    output_shape.push_back(
        (input_dims[i + 2] + paddings[2 * i] + paddings[2 * i + 1] - dkernel) /
            strides[i] +
        1);
  }

  const auto output_shape_nhwc = DimNCHW2NHWC(output_shape);
  const auto output_tensor = graph->AddNode(output_var_name,
                                            output_shape_nhwc,
                                            CNML_TENSOR,
                                            CNML_NHWC,
                                            graph->FPType());
  scope->FindVar(output_var_name)
      ->GetMutable<::paddle::lite::Tensor>()
      ->Resize(output_shape_nhwc);

  // Create filter node
  const auto filter_tensor = graph->AddNode(filter_var_name,
                                            filter_dims.Vectorize(),
                                            CNML_FILTER,
                                            CNML_NCHW,
                                            graph->FPType());
  const auto weight_scale =
      op_info->GetAttr<std::vector<float>>("weight_scale");

  if (filter->precision() == PrecisionType::kUnk ||
      filter->precision() == PrecisionType::kInt8) {
    std::vector<float> filter_dequant(filter->data_size());
    dequant(filter_dequant.data(),
            filter->mutable_data<int8_t>(),
            1,
            filter_dims[0],
            filter_dims[1] * filter_dims[2] * filter_dims[3],
            weight_scale);
    transpose(filter_dequant.data(),
              filter->mutable_data<float>(),
              {static_cast<int>(filter_dims[0]),
               static_cast<int>(filter_dims[1]),
               static_cast<int>(filter_dims[2]),
               static_cast<int>(filter_dims[3])},
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
      bias_shape = {oc};
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
    bias_tensor = graph->AddNode(bias_var_name,
                                 bias_dims.Vectorize(),
                                 CNML_CONST,
                                 CNML_CNHW,
                                 graph->FPType());
    graph->BindConstData(bias_var_name, bias);
  }

  const auto input_scale = op_info->GetAttr<float>("input_scale");

  bool use_first_conv = false;
  if (lite::DeviceInfo::Global().UseFirstConv() && input_dims_nhwc[3] == 3) {
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
                                            std::vector<int64_t>{3},
                                            CNML_CONST,
                                            CNML_CNHW,
                                            graph->FPType());
    const auto std_tensor = graph->AddNode("first_conv_std_tensor",
                                           std::vector<int64_t>{3},
                                           CNML_CONST,
                                           CNML_CNHW,
                                           graph->FPType());

    graph->BindConstRawData("first_conv_mean_tensor",
                            lite::DeviceInfo::Global().MeanVec().data(),
                            3,
                            false);
    graph->BindConstRawData("first_conv_std_tensor",
                            lite::DeviceInfo::Global().StdVec().data(),
                            3,
                            false);

    graph->GetNode(input_var_name)->set_mlu_dtype(CNML_DATA_UINT8);
    CNML_CALL(cnmlCreateConvFirstOpForward(
        &conv_op,
        conv_param,
        graph->GetNode(input_var_name)->mlu_tensor(),
        mean_tensor->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr,
        std_tensor->mlu_tensor()));
    CNML_CALL(cnmlDestroyConvFirstOpParam(&conv_param));
  } else {
    cnmlConvOpParam_t conv_param;
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
        graph->GetNode(input_var_name)->mlu_tensor(),
        output_tensor->mlu_tensor(),
        filter_tensor->mlu_tensor(),
        bias_tensor ? bias_tensor->mlu_tensor() : nullptr));
    CNML_CALL(cnmlDestroyConvOpParam(&conv_param));
  }

  graph->SetComputingDataType(
      conv_op, graph->GetNode(input_var_name)->mlu_tensor(), 1 / input_scale);
  graph->SetComputingDataType(
      conv_op,
      filter_tensor->mlu_tensor(),
      1 / *min_element(weight_scale.begin(), weight_scale.end()));
  CNML_CALL(cnmlSetOperationComputingLayout(conv_op, CNML_NHWC));
  if (HasInputArg(op_info, scope, "Bias")) {
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
    graph->BindConstData(bias_var_name, bias);
  }
  graph->BindConstData(filter_var_name, filter);
  graph->FuseOp(conv_op);
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
