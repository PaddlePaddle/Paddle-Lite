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

#include "lite/kernels/mlu/bridges/test_helper.h"
#include <utility>
#include "lite/core/device_info.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/mlu/subgraph_compute.h"
#include "lite/utils/macros.h"
namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <lite_api::PrecisionType Dtype>
void PrepareInput(Graph* graph,
                  const std::string& input_name,
                  Tensor* input_tensor,
                  cnmlDataOrder_t order) {
  static LITE_THREAD_LOCAL Tensor temp_input;
  temp_input.Resize(input_tensor->dims().Vectorize());
  temp_input.CopyDataFrom(*input_tensor);
  using data_type = typename MLUTypeTraits<Dtype>::type;
  auto input_node = graph->AddNode(
      input_name,
      input_tensor->dims().Vectorize(),
      CNML_TENSOR,
      CNML_NCHW,
      MLUTypeTraits<Dtype>::cnml_type,
      order,
      reinterpret_cast<void*>(
          input_tensor->template mutable_data<data_type>(TARGET(kMLU))));
  CHECK(input_node);
  CNRT_CHECK(cnrtMemcpy(input_tensor->template mutable_data<data_type>(),
                        temp_input.mutable_data<data_type>(),
                        sizeof(data_type) * input_tensor->dims().production(),
                        CNRT_MEM_TRANS_DIR_HOST2DEV));
}

void LaunchOp(const std::shared_ptr<lite::OpLite> op,
              const std::vector<std::string>& input_var_names,
              const std::vector<std::string>& output_var_names,
              cnmlDataOrder_t order) {
  CNRT_CALL(cnrtInit(0));
  lite::SetMluDevice(0);
  cnrtQueue_t queue_;
  CNRT_CALL(cnrtCreateQueue(&queue_));
  cnrtDev_t dev_handle;
  CNRT_CALL(cnrtGetDeviceHandle(&dev_handle, 0));
  CNRT_CALL(cnrtSetCurrentDevice(dev_handle));
  auto scope = op->scope();
  auto op_type = op->op_info()->Type();
  paddle::lite::subgraph::mlu::Graph graph;
  // convert op to IR graph
  const auto& bridges = subgraph::Registry::Instance();
  CHECK(bridges.Exists(op_type, TARGET(kMLU)));

  // Convert input data var and add it into the MLU IR graph
  for (auto& input_name : input_var_names) {
    auto input_tensor = scope->FindMutableTensor(input_name);
    auto data_type = input_tensor->precision();

    switch (data_type) {
#define PREPARE_INPUT(type__)                                                 \
  case PRECISION(type__):                                                     \
    PrepareInput<PRECISION(type__)>(&graph, input_name, input_tensor, order); \
    break;
      PREPARE_INPUT(kFP16)
      PREPARE_INPUT(kFloat)
      PREPARE_INPUT(kInt8)
      PREPARE_INPUT(kInt32)
#undef PREPARE_INPUT
      default:
        CHECK(0);
    }
  }
  op->CheckShape();
  op->InferShape();
  bridges.Select(op_type, TARGET(kMLU))(
      reinterpret_cast<void*>(&graph), const_cast<OpLite*>(op.get()), nullptr);

  for (auto& output_name : output_var_names) {
    if (graph.HasNode(output_name)) {
      graph.AddOutput(graph.GetNode(output_name));
    }
    auto output_tensor = scope->FindMutableTensor(output_name);
    void* p_data =
        static_cast<void*>(output_tensor->mutable_data<float>(TARGET(kMLU)));
    auto node = graph.GetNode(output_name);
    CHECK(p_data);
    node->set_mlu_ptr(p_data);
  }
  for (auto& input_name : input_var_names) {
    graph.AddInput(graph.GetNode(input_name));
  }

  graph.Compile(CNML_MLU270, 1);
  graph.Compute(queue_, *(graph.MutableInputs()), *(graph.MutableOutputs()));
  CNRT_CALL(cnrtSyncQueue(queue_));

  for (auto& output_name : output_var_names) {
    auto output_tensor = scope->FindMutableTensor(output_name);
    Tensor temp_out;
    temp_out.Resize(output_tensor->dims().Vectorize());
    CNRT_CHECK(cnrtMemcpy(temp_out.mutable_data<float>(TARGET(kHost)),
                          output_tensor->mutable_data<float>(),
                          sizeof(float) * output_tensor->dims().production(),
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    output_tensor->mutable_data<float>(TARGET(kHost));
    output_tensor->CopyDataFrom(temp_out);
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

// USE_LITE_OP(graph_op);
// USE_LITE_KERNEL(graph_op, kMLU, kFloat, kNHWC, def);
