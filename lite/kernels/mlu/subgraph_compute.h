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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/core/types.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/tensor.h"
#include "lite/kernels/npu/bridges/engine.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/utils/env.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

template <PrecisionType Precision>
class SubgraphEngine : public subgraph::Engine {
 public:
  SubgraphEngine(KernelContext* ctx,
                 int block_idx,
                 cpp::BlockDesc* block_desc,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 Scope* scope,
                 paddle::lite_api::PrecisionType type)
      : subgraph::Engine(
            ctx, block_idx, block_desc, input_names, output_names, scope),
        fp_type_(type) {
    VLOG(4) << "[MLU] PADDLE_LITE_MLU_SAVE_OFFLINE_MODEL is "
            << GetBoolFromEnv("PADDLE_LITE_MLU_SAVE_OFFLINE_MODEL");
    VLOG(4) << "[MLU] PADDLE_LITE_MLU_DISABLE_BATCH_SIZE_CHANGEABLE is "
            << GetBoolFromEnv("PADDLE_LITE_MLU_DISABLE_BATCH_SIZE_CHANGEABLE");
    if (GetBoolFromEnv("PADDLE_LITE_MLU_DISABLE_BATCH_SIZE_CHANGEABLE")) {
      disable_batch_size_changeable_ = true;
    }
  }

  int Build() {
    // In order to attach all of the ops of the block desc, we need to build
    // the original program firstly.
    BuildOriginProgram();
    // Run InferShape() of all of ops, and convert Paddle ops to MLU IR graph
    build_device_program_status_ = BuildDeviceProgram();
    return build_device_program_status_;
  }

  int Launch() {
    // Rebuild device program when the shapes of input tensors have been
    // changed.
    if (subgraph::CHECK_SUCCESS(build_device_program_status_) &&
        subgraph::CHECK_REBUILD_WHEN_SHAPE_CHANGED(
            build_device_program_status_) &&
        InputShapeChanged()) {
      Build();
    }
    if (subgraph::CHECK_FAILED(build_device_program_status_)) {
      LaunchOriginProgram();
    } else {
      LaunchDeviceProgram();
    }
    return 0;
  }

  bool InputShapeChanged() {
    std::vector<std::vector<int64_t>> new_shape;
    // used in batch changable situation
    std::vector<std::vector<int64_t>> all_shape;
    for (auto origin_itensor : origin_itensors_) {
      if (!disable_batch_size_changeable_) {
        auto iv = origin_itensor->dims().Vectorize();
        all_shape.push_back(iv);
        iv.erase(iv.begin());
        new_shape.push_back(iv);
      } else {
        new_shape.push_back(origin_itensor->dims().Vectorize());
      }
    }
    inputs_shape_ = new_shape;
    all_inputs_shape_ = all_shape;
    if (shape_graph_map_.count(inputs_shape_) > 0) {
      return false;
    }
    return true;
  }

  inline cnmlDataType_t PrecisionToDatatype(PrecisionType data_type) {
    switch (data_type) {
      case paddle::lite_api::PrecisionType::kFP16:
        return CNML_DATA_FLOAT16;
      case paddle::lite_api::PrecisionType::kFloat:
        return CNML_DATA_FLOAT32;
      case paddle::lite_api::PrecisionType::kInt32:
        return CNML_DATA_INT32;
      case paddle::lite_api::PrecisionType::kInt8:
        return CNML_DATA_INT8;
      default:
        return PrecisionToDatatype(fp_type_);
    }
  }

 protected:
  int BuildDeviceProgram() override {
    if (!error_compile_batch_size_changeable_ &&
        !disable_batch_size_changeable_) {
      int status = BuildDeviceProgramImpl();
      if (subgraph::CHECK_SUCCESS(status)) {
        return status;
      }
      VLOG(4) << "[MLU] build batch_size changeable subgraph op failed, "
                 "changed to input_shape changeable";
    }
    error_compile_batch_size_changeable_ = true;
    disable_batch_size_changeable_ = true;
    return BuildDeviceProgramImpl();
  }

  int BuildDeviceProgramImpl() {
    int status = 0;
    auto graph = std::make_shared<paddle::lite::subgraph::mlu::Graph>();
    graph->SetFPType(fp_type_);
    std::vector<std::vector<int64_t>> new_shape;
    origin_itensors_.clear();
    origin_otensors_.clear();

    auto data_order = block_desc_->GetOp<cpp::OpDesc>(0)->Type() == "layout"
                          ? CNML_NCHW
                          : CNML_NHWC;
    // Convert all of input data vars and added into the MLU IR graph
    status |= subgraph::REBUILD_WHEN_SHAPE_CHANGED;
    for (auto& input_name : input_names_) {
      auto input_tensor = scope_->FindMutableTensor(input_name);
      auto data_type = input_tensor->precision();
      cnmlDataType_t fp_type = PrecisionToDatatype(data_type);
      origin_itensors_.push_back(input_tensor);
      if (!disable_batch_size_changeable_) {
        auto iv = input_tensor->dims().Vectorize();
        iv.erase(iv.begin());
        new_shape.push_back(iv);
      } else {
        new_shape.push_back(input_tensor->dims().Vectorize());
      }

      CHECK(input_tensor);
      VLOG(4) << "subgraph input tensor " << input_name << std::endl;
      auto input_node = graph->AddNode(input_name,
                                       input_tensor->dims().Vectorize(),
                                       CNML_TENSOR,
                                       CNML_NCHW,
                                       fp_type,
                                       data_order);
      CHECK(input_node);
      // MLU doesn't support dynamic dimensions/shapes, so need to rebuild
      // the program when the shape of any input tensor is changed.
    }
    LOG(INFO) << "START TO CONVERT ";
    // Convert all of ops and its weights and added into the MLU IR graph
    const auto& bridges = subgraph::Registry::Instance();
    for (auto& inst : origin_program_) {
      auto op = inst.op();
      CHECK(op);
      std::string op_type = op->op_info()->Type();
      // since cnml's compile api will not return error now, we simply check
      // op's type
      if (!disable_batch_size_changeable_ &&
          std::find(unsupport_batch_size_changeable_op_type_.begin(),
                    unsupport_batch_size_changeable_op_type_.end(),
                    op_type) !=
              unsupport_batch_size_changeable_op_type_.end()) {
        status |= subgraph::FAILED;
        VLOG(4) << "[MLU] found unsupported batch_size changeable op type: "
                << op_type;
        return status;
      }
      op->CheckShape();
      const_cast<OpLite*>(op)->InferShape();
      if (!bridges.Exists(op_type, TARGET(kMLU))) {
        LOG(INFO) << "MLU bridges doesn't support op_type: " << op_type;
        return subgraph::FAILED;
      }
      auto kernel = inst.kernel();
      status |= bridges.Select(op_type, TARGET(kMLU))(
          reinterpret_cast<void*>(graph.get()),
          const_cast<OpLite*>(op),
          const_cast<KernelBase*>(kernel));
      if (subgraph::CHECK_FAILED(status)) {
        return subgraph::FAILED;
      }
    }
    // Obtain the output nodes of the MLU IR graph and build the graph to MLU
    // runtime
    for (auto& output_name : output_names_) {
      if (graph->HasNode(output_name)) {
        graph->AddOutput(graph->GetNode(output_name));
        auto output_tensor = scope_->FindMutableTensor(output_name);
        origin_otensors_.push_back(output_tensor);
        VLOG(4) << "subgraph output tensor " << output_name << std::endl;

        // auto node = graph->GetNode(output_name);
        // CHECK(p_data);
        // node->set_mlu_ptr(p_data);
      }
    }
    for (auto& input_name : input_names_) {
      graph->AddInput(graph->GetNode(input_name),
                      disable_batch_size_changeable_);
    }

    CHECK(!origin_otensors_.empty()) << "[MLU] no valid output names";
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto core_version = mlu_context.MLUCoreVersion();
    auto core_number = mlu_context.MLUCoreNumber();
    graph->Compile(core_version, core_number);
    shape_graph_map_[new_shape] = graph;
    if (GetBoolFromEnv("PADDLE_LITE_MLU_SAVE_OFFLINE_MODEL")) {
      graph->GenOfflineModel(GetOfflineModName());
    }
    return status;
  }

  std::string TrimStrings(const std::string& origin_str) {
    std::string str = origin_str;
    std::size_t found = str.find("0x");
    std::size_t found_end = 0;
    const std::vector<std::string> del_strs = {
        "/trans_io_copy", "/trans_cast", "/trans_layout"};
    for (const auto& iterm : del_strs) {
      found_end = str.find(iterm);
      // trim point address and one of the del_strs
      if (found != std::string::npos && found_end != std::string::npos) {
        str.replace(found, found_end - found, "");
        found_end = str.find(iterm);
        str.replace(found_end, iterm.size(), "");
        break;
      }
    }
    return str;
  }

  std::string GetOfflineModName() {
    sort(input_names_.begin(), input_names_.end());
    sort(output_names_.begin(), output_names_.end());
    const auto& delimiter = "__";
    const auto& delimiter_num = "_";
    const auto& input_shape_str = "input_shape_";
    const auto& output_shape_str = "output_shape_";
    std::string name = "";
    std::string tmp = "";
    for (const auto& input_name : input_names_) {
      tmp = input_name;
      name += TrimStrings(tmp) + delimiter + input_shape_str;
      auto input_tensor = scope_->FindMutableTensor(input_name);
      for (const auto& iterm : input_tensor->dims().Vectorize()) {
        name += std::to_string(iterm) + delimiter_num;
      }
      name += delimiter;
    }
    for (const auto& output_name : output_names_) {
      tmp = output_name;
      name += TrimStrings(tmp) + delimiter + output_shape_str;
      auto output_tensor = scope_->FindMutableTensor(output_name);
      for (const auto& iterm : output_tensor->dims().Vectorize()) {
        name += std::to_string(iterm) + delimiter_num;
      }
      name += delimiter;
    }
    std::replace(name.begin(), name.end(), '/', '-');
    return name;
  }

  void InferOutputsShapeOnly() {
    // infer outputs shape when enable BATCH_SIZE_CHANGEABLE
    const auto iter = in_out_shape_map_.find(all_inputs_shape_);
    if (iter != in_out_shape_map_.end()) {
      for (size_t i = 0; i < origin_otensors_.size(); ++i) {
        origin_otensors_[i]->Resize(iter->second[i]);
      }
    } else {
      for (auto& inst : origin_program_) {
        auto op = inst.op();
        CHECK(op);
        op->CheckShape();
        const_cast<OpLite*>(op)->InferShape();
      }
      std::vector<std::vector<int64_t>> outs_shape;
      for (size_t i = 0; i < origin_otensors_.size(); ++i) {
        outs_shape.push_back(origin_otensors_[i]->dims().Vectorize());
      }
      in_out_shape_map_[all_inputs_shape_] = outs_shape;
    }
  }

  inline void* GetOutputDataPtr(Tensor* tensor, bool use_mlu_cast) {
    if (use_mlu_cast) {
      // output is float, since cast fused in subgraph
      return static_cast<void*>(tensor->mutable_data<float>(TARGET(kMLU)));
    } else {
      return static_cast<void*>(
          tensor->template mutable_data<
              typename subgraph::mlu::MLUTypeTraits<Precision>::type>(
              TARGET(kMLU)));
    }
  }

  int LaunchDeviceProgram() override {
    // prepare input and output memory
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto exec_queue = mlu_context.exec_queue();
    u32_t affinity = mlu_context.affinity();
    cnrtInvokeFuncParam_t forward_param = mlu_context.forward_param();
    int data_param = 1;
    forward_param.data_parallelism = &data_param;
    forward_param.affinity = &affinity;
    forward_param.end = CNRT_PARAM_END;

    auto graph = shape_graph_map_[inputs_shape_];
    auto* graph_input = graph->MutableInputs();
    auto* graph_output = graph->MutableOutputs();
    CHECK_EQ(graph_input->size(), origin_itensors_.size());
    CHECK_EQ(graph_output->size(), origin_otensors_.size());

    bool use_mlu_cast = GetBoolFromEnv("LITE_MLU_CAST");

    if (!disable_batch_size_changeable_) {
      std::vector<std::shared_ptr<paddle::lite::subgraph::mlu::MLUTensor>>
          graph_in;
      if (shape_tensor_map_in_.find(all_inputs_shape_) !=
          shape_tensor_map_in_.end()) {
        graph_in = shape_tensor_map_in_[all_inputs_shape_];
        for (size_t i = 0; i < origin_itensors_.size(); ++i) {
          graph_in[i]->set_mlu_ptr(
              const_cast<void*>(origin_itensors_[i]->raw_data()));
        }
      } else {
        graph_in.reserve(origin_itensors_.size());
        for (size_t i = 0; i < origin_itensors_.size(); ++i) {
          paddle::lite::subgraph::mlu::MLUTensor tmp(
              origin_itensors_[i]->dims().Vectorize());
          tmp.set_mlu_dtype(graph_input->at(i)->dtype());
          tmp.set_mlu_ptr(const_cast<void*>(origin_itensors_[i]->raw_data()));
          graph_in.push_back(
              std::make_shared<paddle::lite::subgraph::mlu::MLUTensor>(tmp));
        }
        shape_tensor_map_in_[all_inputs_shape_] = graph_in;
      }

      // TODO(zhangmingwei): we just call every op's infer_shape to get outputs'
      // shape, may be it's better to use cnml's api to get output shape. This
      // can be done when cnml's tensor dimension is totally equal to lite's
      // tensor
      // shape.
      InferOutputsShapeOnly();
      // const std::vector<std::vector<int64_t>> new_output_size =
      //    graph->InferOutputsShape(graph_in);

      std::vector<std::shared_ptr<paddle::lite::subgraph::mlu::MLUTensor>>
          graph_out;

      if (shape_tensor_map_out_.find(all_inputs_shape_) !=
          shape_tensor_map_out_.end()) {
        graph_out = shape_tensor_map_out_[all_inputs_shape_];
        for (size_t i = 0; i < origin_otensors_.size(); ++i) {
          // origin_otensors_[i]->Resize(new_output_size.at(i));
          graph_out[i]->set_mlu_ptr(
              GetOutputDataPtr(origin_otensors_[i], use_mlu_cast));
        }
      } else {
        graph_out.reserve(origin_otensors_.size());
        for (size_t i = 0; i < origin_otensors_.size(); ++i) {
          // origin_otensors_[i]->Resize(new_output_size.at(i));
          paddle::lite::subgraph::mlu::MLUTensor tmp(
              origin_otensors_[i]->dims().Vectorize());
          tmp.set_mlu_dtype(graph_output->at(i)->dtype());
          tmp.set_mlu_ptr(GetOutputDataPtr(origin_otensors_[i], use_mlu_cast));
          graph_out.push_back(
              std::make_shared<paddle::lite::subgraph::mlu::MLUTensor>(tmp));
        }
        shape_tensor_map_out_[all_inputs_shape_] = graph_out;
      }
      graph->Compute(forward_param, exec_queue, graph_in, graph_out);
    } else {
      for (size_t i = 0; i < origin_itensors_.size(); ++i) {
        graph_input->at(i)->set_mlu_ptr(
            const_cast<void*>(origin_itensors_[i]->raw_data()));
      }
      for (size_t i = 0; i < origin_otensors_.size(); ++i) {
        origin_otensors_[i]->Resize(graph_output->at(i)->get_origin_shape());
        graph_output->at(i)->set_mlu_ptr(
            GetOutputDataPtr(origin_otensors_[i], use_mlu_cast));
      }
      graph->Compute(forward_param, exec_queue);
    }

    // // =========== DUMP ===================
    // for (auto input_name : input_names_) {
    //   auto input_tensor =
    //   shape_graph_map_[inputs_shape_]->GetNode(input_name);
    //   auto dump_name = input_name;
    //   while (dump_name.find("/") != std::string::npos) {
    //     dump_name = dump_name.replace(dump_name.find("/"), 1, "_");
    //   }
    //   VLOG(6) << "dump_name: " << dump_name;
    //   input_tensor->ToFile(dump_name);
    // }
    // for (auto output_name : output_names_) {
    //   if (shape_graph_map_[inputs_shape_]->HasNode(output_name)) {
    //     auto output_tensor =
    //     shape_graph_map_[inputs_shape_]->GetNode(output_name);
    //     auto dump_name = output_name;
    //     while (dump_name.find("/") != std::string::npos) {
    //       dump_name = dump_name.replace(dump_name.find("/"), 1, "_");
    //     }
    //     VLOG(6) << "dump_name: " << dump_name;
    //     output_tensor->ToFile(dump_name);
    //   } else {
    //     VLOG(6) << "graph does not have " << output_name << " as output"
    //             << std::endl;
    //   }
    // }
    // // =========== DUMP END ================
    return 0;
  }

  paddle::lite_api::PrecisionType fp_type_;
  std::vector<std::vector<int64_t>> inputs_shape_{};
  std::vector<std::vector<int64_t>> all_inputs_shape_{};
  std::map<std::vector<std::vector<int64_t>>,
           std::shared_ptr<paddle::lite::subgraph::mlu::Graph>>
      shape_graph_map_{};
  // enable batch size changeable by default, this cound be changed by
  // environment variable PADDLE_LITE_MLU_DISABLE_BATCH_SIZE_CHANGEABLE and
  // whether the op can be compiled with batch size changeable way
  bool disable_batch_size_changeable_{false};
  bool error_compile_batch_size_changeable_{false};
  std::vector<std::string> unsupport_batch_size_changeable_op_type_{"concat"};
  // search output runtime MLUTensor for certain output shape when enable
  // BATCH_SIZE_CHANGEABLE
  std::map<std::vector<std::vector<int64_t>>,
           std::vector<std::shared_ptr<paddle::lite::subgraph::mlu::MLUTensor>>>
      shape_tensor_map_out_{};
  // search input runtime MLUTensor for certain input shape when enable
  // BATCH_SIZE_CHANGEABLE
  std::map<std::vector<std::vector<int64_t>>,
           std::vector<std::shared_ptr<paddle::lite::subgraph::mlu::MLUTensor>>>
      shape_tensor_map_in_{};
  // search output shape for certain input shape when enable
  // BATCH_SIZE_CHANGEABLE
  std::map<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
      in_out_shape_map_{};
};

template <PrecisionType Precision>
class SubgraphCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override {
    auto& param = this->template Param<param_t>();
    // LOG(INFO) << "SUBGRAP Prepare RUN index " << param.sub_block_idx;
    engine_.reset(new SubgraphEngine<Precision>(this->ctx_.get(),
                                                param.sub_block_idx,
                                                param.sub_block_desc,
                                                param.input_data_names,
                                                param.output_data_names,
                                                param.scope,
                                                this->precision()));
    CHECK(engine_);
    engine_->Build();
  }

  void Run() override {
    CHECK(engine_);
    engine_->Launch();
  }

  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine<Precision>> engine_;
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
