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

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  subgraph::bm::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  graph.CreateCompilerHandle();
  auto& ctx = this->ctx_->template As<BMContext>();
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kBM))) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |=
        bridges.Select(op_type, TARGET(kBM))(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op),
                                             const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }
  std::string net_name = "bmnetc_f32umodel";
  __bmcompile_opt(
      graph.GetCompilerHandle(), const_cast<char*>(net_name.c_str()), 1);
  void* bmodel_data = nullptr;
  unsigned int data_size = 0;
  bm_hd_ = static_cast<bm_handle_t>(ctx.GetHandle());
  finish_bmcompiler_data(graph.GetCompilerHandle(), &bmodel_data, &data_size);
  bmrt_hd_ = bmrt_create(bm_hd_);
  if (false == bmrt_load_bmodel_data(bmrt_hd_, bmodel_data, data_size)) {
    return subgraph::FAILED;
  }
  bmrt_get_network_names(bmrt_hd_, &net_names_);
  net_info_ = bmrt_get_network_info(bmrt_hd_, net_names_[0]);
  auto& stage = net_info_->stages[0];
  // input
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  device_inputs_.resize(input_names_.size());
  for (size_t i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(net_info_->input_names[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    bm_device_mem_t* p_mem =
        static_cast<bm_device_mem_t*>(malloc(sizeof(bm_device_mem_t)));
    CHECK(p_mem != nullptr);
    CHECK_EQ(bm_malloc_device_byte(
                 bm_hd_, p_mem, origin_itensors_[i]->memory_size()),
             BM_SUCCESS);
    bmrt_tensor_with_device(&device_inputs_[i],
                            *p_mem,
                            net_info_->input_dtypes[i],
                            stage.input_shapes[i]);
  }
  // output
  origin_odims_.resize(output_names_.size());
  origin_otensors_.resize(output_names_.size());
  device_outputs_.resize(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(net_info_->output_names[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    origin_otensors_[i]->mutable_data<float>();
    bm_device_mem_t* p_mem =
        static_cast<bm_device_mem_t*>(malloc(sizeof(bm_device_mem_t)));
    CHECK(p_mem != nullptr);
    CHECK_EQ(bm_malloc_device_byte(
                 bm_hd_, p_mem, origin_otensors_[i]->memory_size()),
             BM_SUCCESS);
    bmrt_tensor_with_device(&device_outputs_[i],
                            *p_mem,
                            net_info_->output_dtypes[i],
                            stage.output_shapes[i]);
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  for (size_t i = 0; i < device_inputs_.size(); i++) {
    bm_memcpy_s2d(bm_hd_,
                  device_inputs_[i].device_mem,
                  const_cast<void*>(origin_itensors_[i]->raw_data()));
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
  for (size_t i = 0; i < device_outputs_.size(); i++) {
    bm_memcpy_d2s(bm_hd_,
                  const_cast<void*>(origin_otensors_[i]->raw_data()),
                  device_outputs_[i].device_mem);
  }
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.sub_block_idx,
                                   param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
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
