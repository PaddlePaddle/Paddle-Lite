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

#include "lite/core/optimizer/optimizer.h"
#include <fstream>
#ifdef LITE_WITH_XPU
#include "lite/core/optimizer/mir/__xpu__static_kernel_pick_pass.h"
#endif
#include "lite/core/optimizer/mir/static_kernel_pick_pass.h"
#include "lite/core/optimizer/mir/type_target_cast_pass.h"
#include "lite/model_parser/model_parser.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {

void Optimizer::AddPass(const std::string& pass_name) {
  mir::Pass* pass = mir::PassManager::Global().LookUp(pass_name);
  passes_.push_back(pass);
}

std::unique_ptr<RuntimeProgram> Optimizer::GenRuntimeProgram(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
  auto pass = mir::PassManager::Global().LookUp<mir::GenerateProgramPass>(
      "generate_program_pass");
  for (auto& graph : *graphs) {
    pass->Apply(graph);
  }
  auto program = pass->GenProgram();
  CHECK(exec_scope_);
  program->set_exec_scope(exec_scope_);
  return program;
}

std::unique_ptr<RuntimeProgram> Optimizer::Run(Program&& program) {
  auto block_size = program.block_size();
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    std::unique_ptr<mir::SSAGraph> graph;
    graph.reset(new mir::SSAGraph);
    graph->Build(program, valid_places_, block_idx);
    graph->SetValidPlaces(valid_places_);
    graphs_.emplace_back(std::move(graph));
  }
  SpecifyKernelPickTactic(kernel_pick_factor_);
  InitTargetTypeTransformPass();
  InitControlFlowOpSharedInputsAndOutputsPlaceSyncPass();

  ApplyPasses(&graphs_);

  exec_scope_ = program.exec_scope();

  return GenRuntimeProgram(&graphs_);
}

void Optimizer::SpecifyKernelPickTactic(core::KernelPickFactor factor) {
  std::string static_pick_name = "static_kernel_pick_pass";
#ifdef LITE_WITH_XPU
  static_pick_name = "__xpu__static_kernel_pick_pass";
#endif
  auto* pass = mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
      static_pick_name);
  CHECK(pass);

  *pass->mutable_kernel_pick_factors() = factor;
}

void Optimizer::InitTargetTypeTransformPass() {
  auto* pass = mir::PassManager::Global().LookUp<mir::TypeTargetTransformPass>(
      "type_target_cast_pass");
  CHECK(pass);
  CHECK(!valid_places_.empty());
  pass->SetValidPlaces(valid_places_);
}

void Optimizer::InitControlFlowOpSharedInputsAndOutputsPlaceSyncPass() {
  auto* pass =
      mir::PassManager::Global()
          .LookUp<mir::ControlFlowOpSharedInputsAndOutputsPlaceSyncPass>(
              "control_flow_op_shared_inputs_and_outputs_place_sync_pass");
  CHECK(pass);
  CHECK(!graphs_.empty());
  pass->SetAllGraphs(&graphs_);
}

void Optimizer::ApplyPasses(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphes) {
  for (auto& pass : passes_) {
    LOG(INFO) << "== Running pass: " << pass->name();
    std::set<TargetType> targets;
    for (const auto& place : valid_places_) {
      targets.insert(place.target);
    }
    bool matched =
        PassMatchesTarget(*pass, targets) && PassMatchesKernels(*pass);
    if (!matched) {
      LOG(INFO) << "   - Skip " << pass->name()
                << " because the target or kernel does not match.";
    } else {
      // Check the pass whether it is supported for processing subblocks
      if (kSubblockUnsupportedPasses.count(pass->name()) ||
          kSubblockSkippedPasses.count(pass->name())) {
        pass->Apply((*graphes)[kRootBlockIdx]);
      } else {
        for (auto& graph : *graphes) {
          pass->Apply(graph);
        }
      }
      LOG(INFO) << "== Finished running: " << pass->name();
    }
  }
}

std::unique_ptr<RuntimeProgram> RunDefaultOptimizer(
    Program&& program,
    const std::vector<Place>& valid_places,
    core::KernelPickFactor kernel_pick_factor,
    const std::vector<std::string>& passes,
    const lite_api::CxxConfig& config) {
  Optimizer optim(valid_places, kernel_pick_factor);

  std::vector<std::string> passes_local{
      {"lite_quant_dequant_fuse_pass",
       "weight_quantization_preprocess_pass",
       "op_transformation_pass",
       "assign_value_calc_offline_pass",
       "p_norm_fill_constant_max_div_fuse_pass",
       "fill_constant_calc_offline_pass",
       "range_calc_offline_pass",
       "scale_calc_offline_pass",
       "unsqueeze_calc_offline_pass",
       "reshape_calc_offline_pass",
       "ssd_boxes_calc_offline_pass",
       // A minimal set of op fusion pass.
       "op_fusion_minimal_set_pass",
       // For the fully quantization model, the quantization parameters of the
       // quantized ops are inferred by the propagation method according to the
       // input scales and out_threashold.
       "quantization_parameters_propagation_pass",
       // Based on the custom mixed precision configuration information, remove
       // the quantization parameters of some quantized ops to force them to run
       // at fp32 precision.
       "quantization_parameters_removal_pass",
       // Subgraph partition based on operator support information defined in
       // lite/kernels/nnadapter/converter/all.h
       "nnadapter_subgraph_pass",
       // Please notify @hong19860320 and @zhupengyang for code review if you
       // want to insert a pass in the above passes.
       "remove_scale1_pass",
       "adaptive_1x1_pool2d_convert_global_pass",  //
       "lite_unsqueeze2_pad3d_squeeze2_fuse_pass",
       "lite_conv_elementwise_fuse_pass",  // conv-elemwise-bn
       "lite_conv_bn_fuse_pass",           //
       "lite_conv_elementwise_fuse_pass",  // conv-bn-elemwise
       "lite_conv_conv_fuse_pass",         //
       // TODO(Superjomn) Refine the fusion related design to select fusion
       // kernels for devices automatically.
       "lite_sigmoid_elementmul_fuse_pass",           //
       "lite_conv_activation_fuse_pass",              //
       "lite_squeeze2_matmul_fuse_pass",              //
       "lite_reshape2_matmul_fuse_pass",              //
       "lite_matmul_element_add_fuse_pass",           //
       "lite_matmul_fuse_pass",                       //
       "lite_fc_fuse_pass",                           //
       "lite_shuffle_channel_fuse_pass",              //
       "lite_transpose_softmax_transpose_fuse_pass",  //
       "lite_interpolate_fuse_pass",                  //
       "identity_scale_eliminate_pass",               //
       "lite_scales_fuse_pass",                       //
       "elementwise_mul_constant_eliminate_pass",     //
       "lite_scale_activation_fuse_pass",             //
       "lite_scaleacts_fuse_pass",                    //
       "lite_elementwise_scale_fuse_pass",            //
       "lite_instance_norm_activation_fuse_pass",     //
       "lite_flatten_fc_fuse_pass",                   //
       "lite_fc_prelu_fuse_pass",                     //
       "lite_elementwise_activation_fuse_pass",
       "lite_conv_scale_fuse_pass",
       "lite_conv_elementwise_tree_fuse_pass",
       "lite_greater_than_cast_fuse_pass",
       "fill_range_fuse_pass",
       "identity_dropout_eliminate_pass",
       "sparse_conv_detect_pass",
       //  "keepdims_convert_pass",
       "__xpu__max_pooling_pad_zero_detect_fuse_pass",
       "__xpu__graph_dedup_pass",
       "__xpu__resnet_fuse_pass",
       "__xpu__conv2d_affine_channel_fuse_pass",
       "__xpu__conv2d_fuse_pass",
       "__xpu__squeeze_excitation_fuse_pass",
       "__xpu__mmdnn_fuse_pass",
       "__xpu__bigru_fuse_pass",
       "__xpu__roformer_relative_pos_fuse_pass",
       "__xpu__quick_gelu_fuse_pass",
       "__xpu__multi_encoder_fuse_pass",
       "__xpu__embedding_with_eltwise_add_fuse_pass",
       "__xpu__fc_fuse_pass",
       "__xpu__softmax_topk_fuse_pass",
       "__xpu__multi_encoder_adaptive_seqlen_fuse_pass",
       "__xpu__multi_encoder_adaptive_seqlen_v2_fuse_pass",
       "__xpu__multi_encoder_slice_link_fuse_pass",
       "__xpu__generate_sequence_fuse_pass",
       "__xpu__logit_fuse_pass",
       "__xpu__link_previous_out_max_pass",
       "fix_mismatched_precision_pass",
       "__xpu__dynamic_lstm_fuse_pass",
       "__xpu__multi_softmax_fuse_pass",
       // pick original kernel from graph (exclude xpu)
       "static_kernel_pick_pass",
       // xpu pick original kernel from graph
       "__xpu__static_kernel_pick_pass",
       "opencl_memory_object_config_pass",
       "remove_tf_redundant_ops_pass",
       // inference arg/var's info(target/precision/layout/device)
       "variable_place_inference_pass",
       "control_flow_op_shared_inputs_and_outputs_place_sync_pass",
       "opencl_kernel_place_correct_pass",
       // debug pass: show arg-type-node's info (target/precision/layout/device)
       "argument_type_display_pass",

       // add io_copy/io_copy_once
       "type_target_cast_pass",
       "variable_place_inference_pass",
       "control_flow_op_shared_inputs_and_outputs_place_sync_pass",
       "argument_type_display_pass",

       "io_copy_kernel_pick_pass",
       "argument_type_display_pass",

       "variable_place_inference_pass",
       "control_flow_op_shared_inputs_and_outputs_place_sync_pass",
       "argument_type_display_pass",

       "type_precision_cast_pass",
       "variable_place_inference_pass",
       "control_flow_op_shared_inputs_and_outputs_place_sync_pass",
       "argument_type_display_pass",

       // add layout/layout_once op
       "type_layout_cast_pass",
       "argument_type_display_pass",

       "variable_place_inference_pass",
       "control_flow_op_shared_inputs_and_outputs_place_sync_pass",
       "argument_type_display_pass",

       "runtime_context_assign_pass",
       "argument_type_display_pass",
       "lite_inplace_fuse_pass",
#ifndef LITE_WITH_PRECISION_PROFILE
       "memory_optimize_pass",
       "xpu_memory_optimize_pass"
#endif
      }};

  // skip the discarded pass
  const std::vector<std::string> discarded_passes =
      config.get_discarded_passes();
  for (auto& pass : discarded_passes) {
    auto iterator = std::find(passes_local.begin(), passes_local.end(), pass);
    if (iterator != passes_local.end()) {
      LOG(INFO) << "discarded pass : " << pass;
      passes_local.erase(iterator);
    } else {
      LOG(INFO) << "the pass : " << pass
                << " dont't exit or has already discarded";
    }
  }

  // It's just a workaround to avoid repeated op fusion if the filter weights
  // are shared among sub-blocks
  if (program.block_size() > 1) {
    passes_local.erase(
        std::remove(
            passes_local.begin(), passes_local.end(), "lite_conv_bn_fuse_pass"),
        passes_local.end());
    // duplicated nodes can't be removed if referenced in different subgraphs
    passes_local.erase(std::remove(passes_local.begin(),
                                   passes_local.end(),
                                   "__xpu__graph_dedup_pass"),
                       passes_local.end());
    LOG(INFO) << "skip __xpu__graph_dedup_pass because of multiple subgraphs["
              << program.block_size() << "]";
  }

  // post_quant_dynamic_pass must be in the behind of
  // lite_quant_dequant_fuse_pass
  const std::string msa_depend_pass{"runtime_context_assign_pass"};
  const std::string pqd_pass{"post_quant_dynamic_pass"};
  const std::string pqd_depend_pass{"lite_quant_dequant_fuse_pass"};
  const std::string fp16_pass{"fp16_attribute_pass"};

  for (const std::string& pass : passes) {
    if (pass == pqd_pass) {
      auto iter =
          std::find(passes_local.begin(), passes_local.end(), pqd_depend_pass);
      CHECK(iter != passes_local.end()) << "No find " << pqd_depend_pass;
      passes_local.push_back(pass);
    } else {
      passes_local.push_back(pass);
    }
  }

  for (auto place : valid_places) {
    if (place.target == TARGET(kARM)) {
      if (place.precision == PRECISION(kFP16)) {
        passes_local.push_back(fp16_pass);
        break;
      }
    }
  }
  for (auto& pass_name : passes_local) {
    optim.AddPass(pass_name);
  }

  return optim.Run(std::move(program));
}

}  // namespace lite
}  // namespace paddle
