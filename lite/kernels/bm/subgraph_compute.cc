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
#include "lite/kernels/bm/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/paddle_use_bridges.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

bool SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  subgraph::bm::Graph graph;
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  graph.CreateCompilerHandle();
  auto& ctx = this->ctx_->template As<BMContext>();
  for (size_t i = 0; i < input_names_.size(); i++) {
    graph.AddNode(input_names_[i]);
  }
  if (!origin_program_) {
    BuildOriginProgram();
  }
  const auto& insts = origin_program_->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    LOG(INFO) << op_type;
    if (!bridges.Exists(op_type, TARGET(kBM))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |=
        bridges.Select(op_type, TARGET(kBM))(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op),
                                             const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }
  std::string net_name = "bmnet_f32bmodel";
  auto unique_net_name = lite::subgraph::bm::UniqueName(net_name);
#ifndef BM_DYNAMIC_COMPILE
  __bmcompile_opt(
      graph.GetCompilerHandle(), const_cast<char*>(unique_net_name.c_str()), 1);
#else
  __bmcompile_ir_opt(
      graph.GetCompilerHandle(), const_cast<char*>(unique_net_name.c_str()), 1);
#endif
  void* bmodel_data = nullptr;
  unsigned int data_size = 0;
  finish_bmcompiler_data(graph.GetCompilerHandle(), &bmodel_data, &data_size);
  graph.UnlockCompilerMutex();
  bm_hd_ = static_cast<bm_handle_t>(ctx.GetHandle());
  if (bmrt_hd_ != nullptr) {
    bmrt_destroy(bmrt_hd_);
    for (size_t i = 0; i < device_inputs_.size(); i++) {
      bmrt_free_device(bmrt_hd_, device_inputs_[i].device_mem);
    }
    for (size_t i = 0; i < device_outputs_.size(); i++) {
      bmrt_free_device(bmrt_hd_, device_outputs_[i].device_mem);
    }
    bmrt_hd_ = nullptr;
  }
  if (bmrt_hd_ == nullptr) {
    bmrt_hd_ = bmrt_create(bm_hd_);
  }
  if (false == bmrt_load_bmodel_data(bmrt_hd_, bmodel_data, data_size)) {
    free(bmodel_data);
    return false;
  }
  free(bmodel_data);
  bmrt_get_network_names(bmrt_hd_, &net_names_);
  net_info_ = bmrt_get_network_info(bmrt_hd_, net_names_[0]);
  auto& stage = net_info_->stages[0];
  // input
  device_inputs_.resize(input_names_.size());
  for (size_t i = 0; i < input_names_.size(); i++) {
    auto origin_itensor =
        exec_scope_->FindMutableTensor(net_info_->input_names[i]);
    CHECK(origin_itensor);
    bm_device_mem_t* p_mem =
        static_cast<bm_device_mem_t*>(malloc(sizeof(bm_device_mem_t)));
    CHECK(p_mem != nullptr);
    CHECK_EQ(
        bmrt_malloc_device_byte(bmrt_hd_, p_mem, origin_itensor->memory_size()),
        0);
    bm_shape_t bmrt_tensor_shape;
    bmrt_tensor_shape.num_dims = origin_itensor->dims().size();
    for (size_t j = 0; j < bmrt_tensor_shape.num_dims; j++) {
      bmrt_tensor_shape.dims[j] = origin_itensor->dims()[j];
    }
    bmrt_tensor_with_device(&device_inputs_[i],
                            *p_mem,
                            net_info_->input_dtypes[i],
                            bmrt_tensor_shape);
    free(p_mem);
  }
  // output
  device_outputs_.resize(net_info_->output_num);
  int out_index = 0;
  for (int i = 0; i < output_names_.size(); i++) {
    outname_map_.insert(std::pair<std::string, int>(output_names_[i], i));
  }
  for (int i = 0; i < net_info_->output_num; i++) {
    auto t_cur = exec_scope_->FindMutableTensor(net_info_->output_names[i]);
    CHECK(t_cur != nullptr);
    bm_device_mem_t* p_mem =
        static_cast<bm_device_mem_t*>(malloc(sizeof(bm_device_mem_t)));
    CHECK(p_mem != nullptr);
    if (outname_map_.find(net_info_->output_names[i]) != outname_map_.end()) {
      origin_otensors_[out_index] = t_cur;
      origin_otensors_[out_index]->mutable_data<float>();
      out_index += 1;
    }
    CHECK_EQ(bmrt_malloc_device_byte(
                 bmrt_hd_, p_mem, net_info_->max_output_bytes[i]),
             0);
    bmrt_tensor_with_device(&device_outputs_[i],
                            *p_mem,
                            net_info_->output_dtypes[i],
                            stage.output_shapes[i]);
    free(p_mem);
  }
  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  for (size_t i = 0; i < device_inputs_.size(); i++) {
    auto origin_itensor =
        exec_scope_->FindMutableTensor(net_info_->input_names[i]);
    device_inputs_[i].shape.dims[0] = origin_itensor->dims()[0];
    bm_memcpy_s2d(bm_hd_,
                  device_inputs_[i].device_mem,
                  const_cast<void*>(origin_itensor->raw_data()));
  }
  bmrt_launch_tensor_ex(bmrt_hd_,
                        net_names_[0],
                        static_cast<const bm_tensor_t*>(&device_inputs_[0]),
                        net_info_->input_num,
                        static_cast<bm_tensor_t*>(&device_outputs_[0]),
                        net_info_->output_num,
                        true,
                        false);
  bm_thread_sync(bm_hd_);
  int out_index = 0;
  for (size_t i = 0; i < device_outputs_.size(); i++) {
    if (outname_map_.find(net_info_->output_names[i]) != outname_map_.end()) {
      bm_tensor_t* bm_otensor = static_cast<bm_tensor_t*>(&device_outputs_[i]);
      std::vector<int64_t> lite_odim;
      for (size_t j = 0; j < bm_otensor->shape.num_dims; j++) {
        lite_odim.push_back(bm_otensor->shape.dims[j]);
      }
      DDim shape_out(lite_odim);
      origin_otensors_[out_index]->Resize(shape_out);
      bm_memcpy_d2s(bm_hd_,
                    const_cast<void*>(origin_otensors_[out_index]->raw_data()),
                    device_outputs_[i].device_mem);
      out_index++;
    }
  }
  return true;
}

bool SubgraphEngine::InputShapeChanged() {
  bool changed = false;
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    auto origin_idim = origin_itensors_[i]->dims().Vectorize();
    if (origin_idims_[i].size() != origin_idim.size()) {
      changed = true;
    } else {
      for (size_t j = 0; j < origin_idim.size(); j++) {
        if (j > 0 && (origin_idim[j] != origin_idims_[i][j])) {
          changed = true;
          break;
        }
      }
    }
    origin_idims_[i] = origin_idim;
  }
  return changed;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names));
  CHECK(engine_);
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Run();
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kBM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::bm::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
