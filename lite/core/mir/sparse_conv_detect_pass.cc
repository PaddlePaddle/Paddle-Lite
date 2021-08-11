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

#include "lite/core/mir/sparse_conv_detect_pass.h"
#include <math.h>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

template <typename T>
int SparseConvDetectPass::ComputeSparseWeight(
    const lite::Tensor* w_tensor,
    const int M,
    const int K,
    const int N,
    const int num_nonzeroes,
    lite::Tensor* nonzero_output_tensor,
    lite::Tensor* oc_nonzeros_tensor,
    lite::Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  int first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {
    oc_nonzeros[ocb] = 0;
    for (int ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[ocb] += 1;
      }
    }
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    diffs[diff_index++] = diff * N;
  }
  return first_ic;
}

template <typename T>
int SparseConvDetectPass::ComputeSparseWeight(const lite::Tensor* w_tensor,
                            const int M,
                            const int K,
                            const int N,
                            const int num_nonzeroes,
                            const int num_build_nonzeroes,
                            lite::Tensor* nonzero_output_tensor,
                            lite::Tensor* oc_nonzeros_tensor,
                            lite::Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  std::vector<int32_t> act_diffs;
  act_diffs.resize(num_nonzeroes);
  int first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {
    oc_nonzeros[ocb] = 0;
    for (int ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          act_diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[ocb] += 1;
      }
    }
    if (oc_nonzeros[ocb] % 4 != 0) {
        int extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
        for (int j = 0; j < extra_zeros; j++) {
            nonzero_output[nonzero_index++] = 0;
        }
    }
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    act_diffs[diff_index++] = diff * N;
  }
  int left_index = 0, right_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {   
      for (int i = 0; i < oc_nonzeros[ocb]; i++) {
          diffs[right_index++] = act_diffs[left_index++];
      }
      if (oc_nonzeros[ocb] % 4 != 0) {
          int extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
          for (int j = 0; j < extra_zeros; j++) {
              diffs[right_index++] = 0;
          }
      }
  }
  return first_ic;
}

template int SparseConvDetectPass::ComputeSparseWeight<float>(
    const lite::Tensor* w_tensor,
    const int M,
    const int K,
    const int N,
    const int num_nonzeroes,
    lite::Tensor* nonzero_output_tensor,
    lite::Tensor* oc_nonzeros_tensor,
    lite::Tensor* diffs_tensor);

template int SparseConvDetectPass::ComputeSparseWeight<int8_t>(
    const lite::Tensor* w_tensor,
    const int M,
    const int K,
    const int N,
    const int num_nonzeroes,
    lite::Tensor* nonzero_output_tensor,
    lite::Tensor* oc_nonzeros_tensor,
    lite::Tensor* diffs_tensor);

template int SparseConvDetectPass::ComputeSparseWeight<float>(
    const lite::Tensor* w_tensor,
    const int M,
    const int K,
    const int N,
    const int num_nonzeroes,
    const int num_build_nonzeroes,
    lite::Tensor* nonzero_output_tensor,
    lite::Tensor* oc_nonzeros_tensor,
    lite::Tensor* diffs_tensor);

template int SparseConvDetectPass::ComputeSparseWeight<int8_t>(
    const lite::Tensor* w_tensor,
    const int M,
    const int K,
    const int N,
    const int num_nonzeroes,
    const int num_build_nonzeroes,
    lite::Tensor* nonzero_output_tensor,
    lite::Tensor* oc_nonzeros_tensor,
    lite::Tensor* diffs_tensor);

template <typename T>
int SparseConvDetectPass::ComputeSparseZeros(const lite::Tensor* weights,
                                             const int num) {
  const T* data = weights->data<T>();
  int zero_num = 0;
  for (int i = 0; i < num; ++i) {
    if (data[i] == static_cast<T>(0)) {
      ++zero_num;
    }
  }
  return zero_num;
}

template <typename T>
int SparseConvDetectPass::ComputeSparseZeros(const lite::Tensor* weights, 
    int* num_build_nonzeroes, const int height, const int width) {
  const T* data = weights->data<T>();
  int num_nonzeroes = 0;
  int num_nonzeroes_act = 0;
  for (int i = 0; i < height; i++) {
    int line_nonzeroes = 0;
    for (int j = 0; j < width; j++) {
      if (data[i * width + j] != static_cast<T>(0)) {
        line_nonzeroes++;
      }
    }
    if (line_nonzeroes % 4 == 0) {
        num_nonzeroes += line_nonzeroes;
    } else {
        num_nonzeroes += line_nonzeroes + 4 - (line_nonzeroes % 4);
    }
    num_nonzeroes_act += line_nonzeroes;
  }
  *num_build_nonzeroes = num_nonzeroes;
  return height * width - num_nonzeroes_act;
}

template int SparseConvDetectPass::ComputeSparseZeros<float>(
    const lite::Tensor* weights, const int num);
template int SparseConvDetectPass::ComputeSparseZeros<int8_t>(
    const lite::Tensor* weights, const int num);

template int SparseConvDetectPass::ComputeSparseZeros<float>(
    const lite::Tensor* weights, int* num_build_nonzeroes, const int height, const int width);
template int SparseConvDetectPass::ComputeSparseZeros<int8_t>(
    const lite::Tensor* weights, int* num_build_nonzeroes, const int height, const int width);

void SparseConvDetectPass::CopyAttrFromOpInfo(cpp::OpDesc* op_desc,
                                              OpInfo* op_info,
                                              const std::string& attr_name) {
  auto attr_type = op_info->GetAttrType(attr_name);
  switch (attr_type) {
    case OpDescAPI::AttrType::INT:
      op_desc->SetAttr(attr_name, op_info->GetAttr<int>(attr_name));
      break;
    case OpDescAPI::AttrType::FLOAT:
      op_desc->SetAttr(attr_name, op_info->GetAttr<float>(attr_name));
      break;
    case OpDescAPI::AttrType::BOOLEAN:
      op_desc->SetAttr(attr_name, op_info->GetAttr<bool>(attr_name));
      break;
    case OpDescAPI::AttrType::STRING:
      op_desc->SetAttr(attr_name, op_info->GetAttr<std::string>(attr_name));
      break;
    case OpDescAPI::AttrType::FLOATS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<float>>(attr_name));
    } break;
    case OpDescAPI::AttrType::INTS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<int>>(attr_name));
    } break;
    case OpDescAPI::AttrType::STRINGS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<std::string>>(attr_name));
    } break;
    default:
      LOG(FATAL) << ":Unknow type(" << static_cast<int>(attr_type) << ")";
      break;
  }
}

void SparseConvDetectPass::CopyInputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                                    OpInfo* op_info,
                                                    const std::string& name) {
  if (op_info->HasInputScale(name, true)) {
    op_desc->SetAttr<std::vector<float>>(name,
                                         op_info->GetInputScale(name, true));
  }
}

void SparseConvDetectPass::CopyOutputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                                     OpInfo* op_info,
                                                     const std::string& name) {
  if (op_info->HasOutputScale(name, true)) {
    op_desc->SetAttr<std::vector<float>>(name,
                                         op_info->GetOutputScale(name, true));
  }
}

void SparseConvDetectPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt() && node->AsStmt().op_type() == "conv2d") {
      auto* scope = node->stmt()->op()->scope();
      auto conv_op_desc = node->stmt()->mutable_op_info();
      auto x = conv_op_desc->Input("Input").front();
      auto w = conv_op_desc->Input("Filter").front();
      auto y = conv_op_desc->Output("Output").front();
      auto x_tensor = scope->FindVar(x)->Get<lite::Tensor>();
      auto w_tensor = scope->FindVar(w)->Get<lite::Tensor>();
      auto x_dims = x_tensor.dims();
      auto weight_dims = w_tensor.dims();
      auto groups = conv_op_desc->GetAttr<int>("groups");
      auto strides = conv_op_desc->GetAttr<std::vector<int>>("strides");
      auto paddings = conv_op_desc->GetAttr<std::vector<int>>("paddings");
      auto ch_out = weight_dims[0];
      auto ch_in = weight_dims[1] * groups;
      auto kh = weight_dims[2];
      auto kw = weight_dims[3];
      auto im_size = x_dims[2] * x_dims[3];
      int weight_num = ch_out * ch_in * kh * kw;
      bool use_int8 = (w_tensor.precision() == PrecisionType::kInt8);
      bool use_fp32 = (w_tensor.precision() == PrecisionType::kFloat);
      if (!(use_int8 || use_fp32)) {
        VLOG(4) << "The sparse conv detect pass now only support fp32 and int8";
        continue;
      }
      if (!(kw == 1 && kh == 1)) {
        VLOG(4) << "The kernel size of the supported sparse conv must be 1x1";
        continue;
      }
      if (groups != 1) {
        VLOG(4) << "The groups of the supported sparse conv must be 1";
        continue;
      }
      if (!(strides[0] == 1 && strides[1] == 1)) {
        VLOG(4) << "The strides of the supported sparse conv must be 1";
        continue;
      }
      if (!(paddings[0] == 0 && paddings[1] == 0)) {
        VLOG(4) << "The paddings of the supported sparse conv must be 0";
        continue;
      }
      int zero_num;
      int num_build_nonzeroes = 0;
      if (use_fp32) {
        // zero_num = ComputeSparseZeros<float>(&w_tensor, weight_num);
        zero_num = ComputeSparseZeros<float>(&w_tensor, &num_build_nonzeroes, ch_out, ch_in);
      } else if (use_int8) {
        zero_num = ComputeSparseZeros<int8_t>(&w_tensor, weight_num);
      }
      int nonzero_num = weight_num - zero_num;
      VLOG(4) << "zero_num: " << zero_num << "weight_num: " << weight_num;
      float sparse_zero_percent =
          static_cast<float>(zero_num) / static_cast<float>(weight_num);
      VLOG(4) << "sparse zero num percent: " << sparse_zero_percent;
      if (sparse_zero_percent < sparse_threshold_) {
        VLOG(4) << "The sparse degree of the sparse conv must be greater than "
                   "sparse_threshold: "
                << sparse_threshold_;
        continue;
      }
      auto nonzeros_output_name =
          string_format("%s_nonzeros_output", w.c_str());
      auto oc_nonzeros_name = string_format("%s_oc_nonzeros", w.c_str());
      auto ic_diffs_name = string_format("%s_ic_diffs", w.c_str());
      auto* nonzeros_output_arg = graph->NewArgumentNode(nonzeros_output_name);
      auto* oc_nonzeros_arg = graph->NewArgumentNode(oc_nonzeros_name);
      auto* ic_diffs_arg = graph->NewArgumentNode(ic_diffs_name);
      nonzeros_output_arg->AsArg().is_persist = true;
      nonzeros_output_arg->AsArg().is_weight = true;
      oc_nonzeros_arg->AsArg().is_persist = true;
      oc_nonzeros_arg->AsArg().is_weight = true;
      ic_diffs_arg->AsArg().is_persist = true;
      ic_diffs_arg->AsArg().is_weight = true;

      auto* nonzeros_output_t =
          scope->Var(nonzeros_output_name)->GetMutable<Tensor>();
      auto* oc_nonzeros_t = scope->Var(oc_nonzeros_name)->GetMutable<Tensor>();
      auto* ic_diffs_t = scope->Var(ic_diffs_name)->GetMutable<Tensor>();
      // nonzeros_output_t->Resize({nonzero_num});
      // oc_nonzeros_t->Resize({ch_out});
      // ic_diffs_t->Resize({nonzero_num});
      if (use_fp32) {
        nonzeros_output_t->Resize({num_build_nonzeroes});
        oc_nonzeros_t->Resize({ch_out});
        ic_diffs_t->Resize({num_build_nonzeroes});
      } else if (use_int8) {
        nonzeros_output_t->Resize({nonzero_num});
        oc_nonzeros_t->Resize({ch_out});
        ic_diffs_t->Resize({nonzero_num});
      }
      int first_ic;
      if (use_fp32) {
        // first_ic = ComputeSparseWeight<float>(&w_tensor,
        //                                       ch_out,
        //                                       ch_in,
        //                                       im_size,
        //                                       nonzero_num,
        //                                       nonzeros_output_t,
        //                                       oc_nonzeros_t,
        //                                       ic_diffs_t);
        first_ic = ComputeSparseWeight<float>(&w_tensor,
                                              ch_out,
                                              ch_in,
                                              im_size,
                                              nonzero_num,
                                              num_build_nonzeroes,
                                              nonzeros_output_t,
                                              oc_nonzeros_t,
                                              ic_diffs_t);
      } else if (use_int8) {
        first_ic = ComputeSparseWeight<int8_t>(&w_tensor,
                                               ch_out,
                                               ch_in,
                                               im_size,
                                               nonzero_num,
                                               nonzeros_output_t,
                                               oc_nonzeros_t,
                                               ic_diffs_t);
      }

      LOG(INFO) << "zero_num: " << zero_num << " weight_num: " << weight_num
                << " first_ic: " << first_ic;
      nonzeros_output_t->set_persistable(true);
      oc_nonzeros_t->set_persistable(true);
      ic_diffs_t->set_persistable(true);
      if (use_fp32)
        nonzeros_output_t->set_precision(PRECISION(kFloat));
      else if (use_int8)
        nonzeros_output_t->set_precision(PRECISION(kInt8));
      oc_nonzeros_t->set_precision(PRECISION(kInt32));
      ic_diffs_t->set_precision(PRECISION(kInt32));
      auto sparse_conv2d_op = LiteOpRegistry::Global().Create("sparse_conv2d");
      cpp::OpDesc op_desc;
      op_desc.SetType("sparse_conv2d");
      op_desc.SetInput("Input", {x});
      op_desc.SetInput("NonZeroWeights", {nonzeros_output_name});
      op_desc.SetInput("OcNonZeros", {oc_nonzeros_name});
      op_desc.SetInput("Diffs", {ic_diffs_name});
      bool has_bias = conv_op_desc->HasInput("Bias") &&
                      conv_op_desc->Input("Bias").size() > 0;
      if (has_bias) {
        auto b = conv_op_desc->Input("Bias").front();
        op_desc.SetInput("Bias", {b});
      }
      op_desc.SetOutput("Output", {y});
      if (use_int8) {
        if (!(conv_op_desc->HasAttr("enable_int8")))
          conv_op_desc->SetAttr<bool>("enable_int8", true);
        else if (conv_op_desc->GetAttr<bool>("enable_int8") == false)
          conv_op_desc->SetAttr<bool>("enable_int8", true);
      }
      // copy attributes
      std::vector<std::string> attr_names = conv_op_desc->AttrNames();
      for (size_t i = 0; i < attr_names.size(); i++) {
        if (conv_op_desc->HasAttr(attr_names[i])) {
          CopyAttrFromOpInfo(&op_desc, conv_op_desc, attr_names[i]);
        }
      }
      // Copy inputs/outputs scales
      if (conv_op_desc->HasAttr("enable_int8")) {
        CopyInputScaleFromOpInfo(&op_desc, conv_op_desc, "Input0_scale");
        CopyInputScaleFromOpInfo(&op_desc, conv_op_desc, "Filter0_scale");
        CopyOutputScaleFromOpInfo(&op_desc, conv_op_desc, "Output0_scale");
      }

      op_desc.SetAttr<int>("first_ic", first_ic);
      sparse_conv2d_op->Attach(op_desc, node->stmt()->op()->scope());
      auto* sparse_op_node = graph->GraphCreateInstructNode(
          sparse_conv2d_op, graph->valid_places());
      for (auto iter = node->inlinks.begin(); iter != node->inlinks.end();) {
        auto it =
            std::find((*iter)->outlinks.begin(), (*iter)->outlinks.end(), node);
        if (it != (*iter)->outlinks.end()) {
          (*iter)->outlinks.erase(it);
        }
        bool is_weight = (*iter)->IsArg() && (*iter)->AsArg().is_weight;
        if (!is_weight) {
          DirectedLink(*iter, sparse_op_node);
        } else {
          graph->RemoveNode((*iter));
        }
        iter = node->inlinks.erase(iter);
      }
      DirectedLink(nonzeros_output_arg, sparse_op_node);
      DirectedLink(oc_nonzeros_arg, sparse_op_node);
      DirectedLink(ic_diffs_arg, sparse_op_node);
      for (auto iter = node->outlinks.begin(); iter != node->outlinks.end();) {
        DirectedLink(sparse_op_node, *iter);
        auto it =
            std::find((*iter)->inlinks.begin(), (*iter)->inlinks.end(), node);
        if (it != (*iter)->inlinks.end()) {
          (*iter)->inlinks.erase(it);
        }
        iter = node->outlinks.erase(iter);
      }
      graph->RemoveNode(node);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(sparse_conv_detect_pass,
                  paddle::lite::mir::SparseConvDetectPass)
    .BindTargets({TARGET(kARM)})
    .ExcludeTargets({TARGET(kXPU)})
    .ExcludeTargets({TARGET(kBM)})
    .ExcludeTargets({TARGET(kRKNPU)})
    .ExcludeTargets({TARGET(kOpenCL)})
    .ExcludeTargets({TARGET(kNPU)})
    .ExcludeTargets({TARGET(kAPU)})
    .ExcludeTargets({TARGET(kHuaweiAscendNPU)})
    .ExcludeTargets({TARGET(kX86)})
    .ExcludeTargets({TARGET(kImaginationNNA)});
