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

#include "lite/core/mir/post_quant_dynamic_pass.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

const std::vector<std::string> PostQuantDynamicPass::quant_axis1_ops{"mul"};

static bool abs_compare(float a, float b) {
  return std::fabs(a) < std::fabs(b);
}

void FindAbsMaxPerChannel(const Tensor& tensor,
                          int quant_axis,
                          std::vector<float>* res) {
  const DDim dims = tensor.dims();
  CHECK(dims.size() == 2 || dims.size() == 4);
  CHECK(tensor.precision() == PrecisionType::kFloat);
  CHECK(quant_axis == 0 || quant_axis == 1);
  CHECK(res);

  res->clear();
  const float* data = tensor.data<float>();

  if (quant_axis == 0) {
    int64_t channel = dims[0];
    int64_t channel_size = dims.production() / channel;
    for (int64_t i = 0; i < channel; i++) {
      const float* start = data + i * channel_size;
      const float* end = start + channel_size;
      const float* iter = std::max_element(start, end, abs_compare);
      res->push_back(std::abs(*iter));
    }
  } else if (quant_axis == 1) {
    int64_t out_size = dims[0];
    int64_t channel = dims[1];
    int64_t inner_size = dims.production() / (out_size * channel);
    for (int64_t i = 0; i < channel; i++) {
      float abs_max = -1;
      for (int64_t j = 0; j < out_size; j++) {
        const float* start = data + j * channel * inner_size + i * inner_size;
        const float* end = start + inner_size;
        const float* iter = std::max_element(start, end, abs_compare);
        float tmp = std::fabs(*iter);
        abs_max = tmp > abs_max ? tmp : abs_max;
      }
      res->push_back(abs_max);
    }
  }
}

template <typename T>
void QuantizeWeightPerChannel(const Tensor& src,
                              const std::vector<float>& scales,
                              int quant_axis,
                              T* dest_data) {
  CHECK(quant_axis == 0 || quant_axis == 1);
  CHECK(dest_data != nullptr);

  const DDim dims = src.dims();
  const float* src_data = src.data<float>();
  if (quant_axis == 0) {
    int64_t channel = dims[0];
    int64_t channel_size = dims.production() / channel;
    for (int64_t i = 0; i < channel; i++) {
      float scale = scales[i];
      const float* src_start = src_data + i * channel_size;
      const float* src_end = src_data + (i + 1) * channel_size;
      T* dest_start = dest_data + i * channel_size;
      std::transform(src_start, src_end, dest_start, [scale](float x) {
        return static_cast<T>(round(x / scale));
      });
    }
  } else if (quant_axis == 1) {
    int64_t out_size = dims[0];
    int64_t channel = dims[1];
    int64_t inner_size = dims.production() / (out_size * channel);
    for (int64_t i = 0; i < out_size; i++) {
      for (int64_t j = 0; j < channel; j++) {
        float scale = scales[j];
        int64_t index = i * channel * inner_size + j * inner_size;
        const float* src_start = src_data + index;
        const float* src_end = src_start + inner_size;
        T* dest_start = dest_data + index;
        std::transform(src_start, src_end, dest_start, [scale](float x) {
          return static_cast<T>(std::round(x / scale));
        });
      }
    }
  }
}

void PostQuantDynamicPerChannel(OpInfo* op_info,
                                Tensor* weight,
                                const std::string weight_name,
                                int quant_axis,
                                int quant_bits) {
  const DDim weight_dims = weight->dims();
  CHECK(weight_dims.size() == 2 || weight_dims.size() == 4);
  CHECK(quant_axis == 0 || quant_axis == 1);
  CHECK(quant_bits == 8 || quant_bits == 16);

  // get scales
  float range = (1 << (quant_bits - 1)) - 1;
  std::vector<float> scales;
  FindAbsMaxPerChannel(*weight, quant_axis, &scales);
  std::transform(
      scales.begin(), scales.end(), scales.begin(), [&range](float x) {
        return x / range;
      });

  // quantize weights
  Tensor tmp_tensor;
  tmp_tensor.CopyDataFrom(*weight);
  weight->clear();

  if (quant_bits == 8) {
    weight->set_precision(PRECISION(kInt8));
    int8_t* weight_data = weight->mutable_data<int8_t>();
    QuantizeWeightPerChannel(tmp_tensor, scales, quant_axis, weight_data);
  } else if (quant_bits == 16) {
    weight->set_precision(PRECISION(kInt16));
    int16_t* weight_data = weight->mutable_data<int16_t>();
    QuantizeWeightPerChannel(tmp_tensor, scales, quant_axis, weight_data);
  }
  op_info->SetAttr<std::string>("quantization_type",
                                "post_weight_channel_wise_abs_max");
  op_info->SetAttr("quantize_weight_bits", quant_bits);
  op_info->SetAttr(weight_name + "_quant_scale", scales);
}

void PostQuantDynamicPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  int quant_bits = 16;
  if (quant_type_ == lite_api::QuantType::QUANT_INT8) {
    quant_bits = 8;
  } else if (quant_type_ == lite_api::QuantType::QUANT_INT16) {
    quant_bits = 16;
  } else {
    LOG(FATAL) << "Not support quant type:" << static_cast<int>(quant_type_);
  }

  std::vector<mir::Node*> nodes;
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt()) {
      const std::string op_type = node->stmt()->op_type();
      auto iter = std::find(quant_ops_.begin(), quant_ops_.end(), op_type);
      if (iter != quant_ops_.end()) {
        nodes.push_back(node);
      }
    }
  }

  for (auto* node : nodes) {
    const std::string op_type = node->stmt()->op_type();
    OpInfo* op_info = node->stmt()->mutable_op_info();
    auto* scope = node->stmt()->op()->scope();
    for (auto* in_node : node->inlinks) {
      CHECK(in_node->IsArg()) << "The input node should be variable.";
      if (in_node->arg()->is_weight) {
        std::string weight_name = in_node->arg()->name;
        Tensor* weight = scope->FindVar(weight_name)->GetMutable<Tensor>();
        CHECK(weight) << "Can not find the weight in scope.";
        if (weight->precision() != PrecisionType::kFloat) {
          LOG(INFO) << "The dtype of weight is not fp32, "
                    << "so skip quantizing the weight of " << weight_name;
          continue;
        }
        auto iter =
            std::find(quant_axis1_ops.begin(), quant_axis1_ops.end(), op_type);
        int quant_axis = iter != quant_axis1_ops.end() ? 1 : 0;
        PostQuantDynamicPerChannel(
            op_info, weight, weight_name, quant_axis, quant_bits);
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(post_quant_dynamic_pass,
                  paddle::lite::mir::PostQuantDynamicPass)
    .BindTargets({TARGET(kAny)});
