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

#include "utility/debug.h"
#include <map>
#include <set>
#include <vector>
#include "driver/nvidia_tensorrt/operation/type.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/string.h"

namespace nnadapter {

static size_t dot_node_counter{0};

/*
 * A Dot template that helps to build a DOT graph definition.
 */
class Dot {
 public:
  struct Attr {
    std::string key;
    std::string value;

    Attr(const std::string& key, const std::string& value)
        : key(key), value(value) {}

    std::string repr() const {
      std::stringstream ss;
      ss << key << "=" << '"' << value << '"';
      return ss.str();
    }
  };

  struct Node {
    std::string name;
    std::vector<Attr> attrs;

    Node(const std::string& name, const std::vector<Attr>& attrs)
        : name(name), attrs(attrs) {
      std::stringstream ss;
      ss << "node_" << dot_node_counter++;
      id_ = ss.str();
    }

    std::string id() const { return id_; }

    std::string repr() const {
      std::stringstream ss;
      NNADAPTER_CHECK(!name.empty());
      ss << id_;
      if (attrs.empty()) {
        ss << "[label=" << '"' << name << '"' << "]";
        return ss.str();
      }
      for (size_t i = 0; i < attrs.size(); i++) {
        if (i == 0) {
          ss << "[label=" << '"' << name << '"' << " ";
        }
        ss << attrs[i].repr();
        ss << ((i < attrs.size() - 1) ? " " : "]");
      }
      return ss.str();
    }

   private:
    std::string id_;
  };

  struct Edge {
    std::string source;
    std::string target;
    std::vector<Attr> attrs;

    Edge(const std::string& source,
         const std::string& target,
         const std::vector<Attr>& attrs)
        : source(source), target(target), attrs(attrs) {}

    std::string repr() const {
      std::stringstream ss;
      NNADAPTER_CHECK(!source.empty());
      NNADAPTER_CHECK(!target.empty());
      ss << source << "->" << target;
      for (size_t i = 0; i < attrs.size(); i++) {
        if (i == 0) {
          ss << "[";
        }
        ss << attrs[i].repr();
        ss << ((i < attrs.size() - 1) ? " " : "]");
      }
      return ss.str();
    }
  };

  Dot() = default;

  explicit Dot(const std::vector<Attr>& attrs) : attrs_(attrs) {}

  void AddNode(const std::string& id,
               const std::vector<Attr>& attrs,
               std::string label = "") {
    NNADAPTER_CHECK(!nodes_.count(id)) << "duplicate Node '" << id << "'";
    if (label.empty()) label = id;
    nodes_.emplace(id, Node{label, attrs});
  }

  void AddEdge(const std::string& source,
               const std::string& target,
               const std::vector<Attr>& attrs) {
    NNADAPTER_CHECK(!source.empty());
    NNADAPTER_CHECK(!target.empty());
    auto sid = nodes_.at(source).id();
    auto tid = nodes_.at(target).id();
    edges_.emplace_back(sid, tid, attrs);
  }

  // Compile to DOT language codes.
  std::string Build() const {
    std::stringstream ss;
    const std::string indent = "   ";
    ss << "digraph G {" << '\n';

    // Add graph attrs
    for (const auto& attr : attrs_) {
      ss << indent << attr.repr() << '\n';
    }
    // add nodes
    for (auto& item : nodes_) {
      ss << indent << item.second.repr() << '\n';
    }
    // add edges
    for (auto& edge : edges_) {
      ss << indent << edge.repr() << '\n';
    }
    ss << "} // end G";
    return ss.str();
  }

 private:
  std::map<std::string, Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Attr> attrs_;
};

NNADAPTER_EXPORT std::string Visualize(core::Model* model) {
#define APPEND_OPERAND_NODE(mode)                                           \
  auto operand_id = OperandIdToString(operand);                             \
  std::string operand_label("nullptr");                                     \
  if (operand != nullptr) {                                                 \
    operand_label = OperandValueToString(operand);                          \
  }                                                                         \
  if (!visited_operands.count(operand)) {                                   \
    dot.AddNode(operand_id, {}, operand_label);                             \
    visited_operands.insert(operand);                                       \
  }                                                                         \
  std::vector<Dot::Attr> attrs;                                             \
  auto& attr_args = mode ? output_args : input_args;                        \
  std::string attr_label = i < attr_args.size() ? attr_args[i] : "unknown"; \
  attrs.emplace_back("label", string_format("%d:%s", i, attr_label.c_str()));

  Dot dot;
  std::ostringstream os;
  auto operations = SortOperationsInTopologicalOrder(model);
  std::set<core::Operand*> visited_operands;
  for (auto* operation : operations) {
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    auto input_count = input_operands.size();
    auto output_count = output_operands.size();
    std::string operation_id = OperationIdToString(operation);
    std::string operation_label = OperationTypeToString(operation->type);
    dot.AddNode(operation_id,
                {Dot::Attr("shape", "box"),
                 Dot::Attr("style", "filled"),
                 Dot::Attr("color", "black"),
                 Dot::Attr("fillcolor", "yellow")},
                operation_label);
    std::vector<std::string> input_args, output_args;
    switch (operation->type) {
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_MAX:
      case NNADAPTER_MIN:
      case NNADAPTER_MUL:
      case NNADAPTER_POW:
      case NNADAPTER_SUB:
        input_args = {"input0", "input1", "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
        input_args = {"input",
                      "auto_pad",
                      "pads",
                      "kernel_shape",
                      "strides",
                      "ceil_mode",
                      "count_include_pad",
                      "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_MAX_POOL_2D:
        input_args = {"input",
                      "auto_pad",
                      "pads",
                      "kernel_shape",
                      "strides",
                      "ceil_mode",
                      "return_indices",
                      "return_indices_dtype",
                      "fuse_code"};
        output_args = {"output", "indices"};
        break;
      case NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D:
        input_args = {"input", "output_shape"};
        output_args = {"output"};
        break;
      case NNADAPTER_ADAPTIVE_MAX_POOL_2D:
        input_args = {
            "input", "output_shape", "return_indices", "return_indices_dtype"};
        output_args = {"output", "indices"};
        break;
      case NNADAPTER_CONCAT:
      case NNADAPTER_STACK:
        input_args.resize(input_count);
        for (int i = 0; i < input_count - 1; i++) {
          input_args[i] = string_format("input%d", i);
        }
        input_args[input_count - 1] = "axis";
        output_args = {"output"};
        break;
      case NNADAPTER_CHANNEL_SHUFFLE:
        input_args = {"input", "group"};
        output_args = {"output"};
        break;
      case NNADAPTER_CONV_2D:
        input_args = {"input",
                      "filter",
                      "bias",
                      "auto_pad",
                      "pads",
                      "strides",
                      "group",
                      "dilations",
                      "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_CONV_2D_TRANSPOSE:
        input_args = {"input",
                      "filter",
                      "bias",
                      "auto_pad",
                      "pads",
                      "strides",
                      "group",
                      "dilations",
                      "output_padding",
                      "output_shape",
                      "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_MAT_MUL:
        input_args = {"x", "y", "transpose_x", "transpose_y"};
        output_args = {"output"};
        break;
      case NNADAPTER_FULLY_CONNECTED:
        input_args = {"input", "weight", "bias", "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_FILL:
        input_args = {"shape", "value"};
        output_args = {"output"};
        break;
      case NNADAPTER_FILL_LIKE:
        input_args = {"input", "value"};
        output_args = {"output"};
        break;
      case NNADAPTER_FLATTEN:
        input_args = {"input", "start_axis", "end_axis"};
        output_args = {"output"};
        break;
      case NNADAPTER_ABS:
      case NNADAPTER_EXP:
      case NNADAPTER_FLOOR:
      case NNADAPTER_LOG:
      case NNADAPTER_NOT:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SQUARE:
      case NNADAPTER_SWISH:
      case NNADAPTER_TANH:
        input_args = {"input"};
        output_args = {"output"};
        break;
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
        input_args = {"input", "alpha", "beta"};
        output_args = {"output"};
        break;
      case NNADAPTER_LEAKY_RELU:
        input_args = {"input", "alpha"};
        output_args = {"output"};
        break;
      case NNADAPTER_PRELU:
        input_args = {"input", "slope"};
        output_args = {"output"};
        break;
      case NNADAPTER_SOFTPLUS:
        input_args = {"input", "beta", "threshold"};
        output_args = {"output"};
        break;
      case NNADAPTER_SLICE:
        input_args = {"input", "axes", "start", "ends", "steps"};
        output_args = {"output"};
        break;
      case NNADAPTER_CLIP:
        input_args = {"input", "min", "max"};
        output_args = {"output"};
        break;
      case NNADAPTER_REDUCE_MEAN:
      case NNADAPTER_REDUCE_MAX:
      case NNADAPTER_REDUCE_SUM:
        input_args = {"input", "axes", "keep_dim"};
        output_args = {"output"};
        break;
      case NNADAPTER_EXPAND:
        input_args = {"input", "shape"};
        output_args = {"output"};
        break;
      case NNADAPTER_RESHAPE:
        input_args = {"input", "shape"};
        output_args = {"output"};
        break;
      case NNADAPTER_RESIZE_NEAREST:
        input_args = {"input", "shape", "scales", "align_corners"};
        output_args = {"output"};
        break;
      case NNADAPTER_RESIZE_LINEAR:
        input_args = {
            "input", "shape", "scales", "align_corners", "align_mode"};
        output_args = {"output"};
        break;
      case NNADAPTER_SOFTMAX:
      case NNADAPTER_LOG_SOFTMAX:
        input_args = {"input", "axis"};
        output_args = {"output"};
        break;
      case NNADAPTER_QUANTIZE:
        input_args = {"input", "axis", "scale", "zero_point"};
        output_args = {"output"};
        break;
      case NNADAPTER_DEQUANTIZE:
        input_args = {"input"};
        output_args = {"output"};
        break;
      case NNADAPTER_CUM_SUM:
        input_args = {"input", "axis", "exclusive", "reverse"};
        output_args = {"output"};
        break;
      case NNADAPTER_GATHER:
        input_args = {"input", "indices", "axis"};
        output_args = {"output"};
        break;
      case NNADAPTER_TOP_K:
        input_args = {
            "input", "k", "axis", "largest", "sorted", "return_indices_type"};
        output_args = {"output", "indices"};
        break;
      case NNADAPTER_ARG_MAX:
      case NNADAPTER_ARG_MIN:
        input_args = {"input", "axis", "keepdim", "dtype"};
        output_args = {"output"};
        break;
      case NNADAPTER_SPLIT:
        input_args = {"input", "axis", "split"};
        output_args.resize(output_count);
        for (size_t i = 0; i < output_count; i++) {
          output_args[i] = string_format("output%d", i);
        }
        break;
      case NNADAPTER_UNSTACK:
        input_args = {"input", "axis", "num"};
        output_args.resize(output_count);
        for (size_t i = 0; i < output_count; i++) {
          output_args[i] = string_format("output%d", i);
        }
        break;
      case NNADAPTER_TRANSPOSE:
        input_args = {"input", "perm"};
        output_args = {"output"};
        break;
      case NNADAPTER_CAST:
        input_args = {"input", "dtype"};
        output_args = {"output"};
        break;
      case NNADAPTER_SHAPE:
        input_args = {"input", "dtype"};
        output_args = {"output"};
        break;
      case NNADAPTER_SQUEEZE:
      case NNADAPTER_UNSQUEEZE:
        input_args = {"input", "axes"};
        output_args = {"output"};
        break;
      case NNADAPTER_ASSIGN:
        input_args = {"input"};
        output_args = {"output"};
        break;
      case NNADAPTER_LP_NORMALIZATION:
        input_args = {"input", "axis", "p", "epsilon"};
        output_args = {"output"};
        break;
      case NNADAPTER_RANGE:
        input_args = {"start", "ends", "step"};
        output_args = {"output"};
        break;
      case NNADAPTER_BATCH_NORMALIZATION:
        input_args = {"input", "scale", "bias", "mean", "variance", "epsilon"};
        output_args = {"output"};
        break;
      case NNADAPTER_INSTANCE_NORMALIZATION:
        input_args = {"input", "scale", "bias", "episilon"};
        output_args = {"output"};
        break;
      case NNADAPTER_LAYER_NORMALIZATION:
        input_args = {"input", "scale", "bias", "begin_norm_axis", "episilon"};
        output_args = {"output"};
        break;
      case NNADAPTER_GROUP_NORMALIZATION:
        input_args = {"input", "scale", "bias", "episilon", "groups"};
        output_args = {"output"};
        break;
      case NNADAPTER_DEFORMABLE_CONV_2D:
        input_args = {"input",
                      "offset",
                      "mask",
                      "filter",
                      "bias",
                      "pads",
                      "strides",
                      "group",
                      "deformable_group",
                      "dilations",
                      "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_PAD:
        input_args = {"input", "pads", "mode", "value"};
        output_args = {"output"};
        break;
      case NNADAPTER_GELU:
        input_args = {"input", "approximate"};
        output_args = {"output"};
        break;
      case NNADAPTER_AND:
      case NNADAPTER_EQUAL:
      case NNADAPTER_GREATER:
      case NNADAPTER_GREATER_EQUAL:
      case NNADAPTER_LESS:
      case NNADAPTER_LESS_EQUAL:
      case NNADAPTER_NOT_EQUAL:
      case NNADAPTER_OR:
      case NNADAPTER_XOR:
        input_args = {"input0", "input1"};
        output_args = {"output"};
        break;
      case NNADAPTER_MESHGRID:
        input_args.resize(input_count);
        for (size_t i = 0; i < input_count; i++) {
          input_args[i] = string_format("input%d", i);
        }
        output_args.resize(output_count);
        for (size_t i = 0; i < output_count; i++) {
          output_args[i] = string_format("output%d", i);
        }
        break;
      case NNADAPTER_TILE:
        input_args = {"input", "repeats"};
        output_args = {"output"};
        break;
      case NNADAPTER_SUM:
        input_args.resize(input_count);
        for (size_t i = 0; i < input_count; i++) {
          input_args[i] = string_format("input%d", i);
        }
        output_args = {"output"};
        break;
      case NNADAPTER_GRID_SAMPLE:
        input_args = {"input", "grid", "aligned_corners", "mode", "pad_pad"};
        output_args = {"output"};
        break;
      case NNADAPTER_ROI_ALIGN:
        input_args = {"input",
                      "rois",
                      "batch_indices",
                      "output_height",
                      "output_width",
                      "sampling_ratio",
                      "spatial_scale",
                      "aligned"};
        output_args = {"output"};
        break;
      case NNADAPTER_WHERE:
        input_args = {"condition", "input0", "input1"};
        output_args = {"output"};
        break;
      case NNADAPTER_YOLO_BOX:
        input_args = {
            "input",
            "imgsize",
            "anchors",
            "class_num",
            "conf_thresh",
            "downsample_ratio",
            "clip_bbox",
            "scale_x_y",
            "iou_aware",
            "iou_aware_factor",
        };
        output_args = {"boxes", "scores"};
        break;
      case NNADAPTER_PRIOR_BOX:
        input_args = {"Input",
                      "Image",
                      "min_sizes",
                      "max_sizes",
                      "aspect_ratios",
                      "variances",
                      "flip",
                      "clip",
                      "step_w",
                      "step_h",
                      "offset",
                      "min_max_aspect_ratios_order"};
        output_args = {"Boxes", "Variances"};
        break;
      case NNADAPTER_NON_MAX_SUPPRESSION:
        input_args = {"BBoxes",
                      "Scores",
                      "background_label",
                      "keep_top_k",
                      "nms_eta",
                      "nms_threshold",
                      "nms_top_k",
                      "normalized",
                      "score_threshold"};
        output_args = {"Out", "NmsRoisNum", "Index"};
        break;
      default:
        if (operation->type < 0) {
          input_args.resize(input_count);
          for (int i = 0; i < input_count; i++) {
            input_args[i] = string_format("input%d", i);
          }
          output_args.resize(output_count);
          for (int i = 0; i < output_count; i++) {
            output_args[i] = string_format("output%d", i);
          }
        } else {
          NNADAPTER_LOG(FATAL) << "unsupported op: "
                               << static_cast<int>(operation->type);
        }
        break;
    }
    for (size_t i = 0; i < input_count; i++) {
      auto* operand = input_operands[i];
      APPEND_OPERAND_NODE(0)
      dot.AddEdge(operand_id, operation_id, attrs);
    }
    for (size_t i = 0; i < output_count; i++) {
      auto* operand = output_operands[i];
      APPEND_OPERAND_NODE(1)
      dot.AddEdge(operation_id, operand_id, attrs);
    }
  }
  os << dot.Build();

#undef APPEND_OPERAND_NODE
  return os.str();
}

#define NNADAPTER_TYPE_TO_STRING(type) \
  case NNADAPTER_##type:               \
    name = #type;                      \
    break;

NNADAPTER_EXPORT std::string ResultCodeToString(NNAdapterResultCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(NO_ERROR);
    NNADAPTER_TYPE_TO_STRING(OUT_OF_MEMORY);
    NNADAPTER_TYPE_TO_STRING(INVALID_PARAMETER);
    NNADAPTER_TYPE_TO_STRING(DEVICE_NOT_FOUND);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandPrecisionCodeToString(
    NNAdapterOperandPrecisionCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(BOOL8);
    NNADAPTER_TYPE_TO_STRING(INT8);
    NNADAPTER_TYPE_TO_STRING(UINT8);
    NNADAPTER_TYPE_TO_STRING(INT16);
    NNADAPTER_TYPE_TO_STRING(UINT16);
    NNADAPTER_TYPE_TO_STRING(INT32);
    NNADAPTER_TYPE_TO_STRING(UINT32);
    NNADAPTER_TYPE_TO_STRING(INT64);
    NNADAPTER_TYPE_TO_STRING(UINT64);
    NNADAPTER_TYPE_TO_STRING(FLOAT16);
    NNADAPTER_TYPE_TO_STRING(FLOAT32);
    NNADAPTER_TYPE_TO_STRING(FLOAT64);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT8_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT8_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT8_ASYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT16_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT16_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT16_ASYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT32_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT32_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT32_ASYMM_PER_LAYER);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandLayoutCodeToString(
    NNAdapterOperandLayoutCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(NCHW);
    NNADAPTER_TYPE_TO_STRING(NHWC);
    NNADAPTER_TYPE_TO_STRING(HWCN);
    NNADAPTER_TYPE_TO_STRING(HWNC);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperandLifetimeCodeToString(
    NNAdapterOperandLifetimeCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(TEMPORARY_VARIABLE);
    NNADAPTER_TYPE_TO_STRING(TEMPORARY_SHAPE);
    NNADAPTER_TYPE_TO_STRING(CONSTANT_COPY);
    NNADAPTER_TYPE_TO_STRING(CONSTANT_REFERENCE);
    NNADAPTER_TYPE_TO_STRING(MODEL_INPUT);
    NNADAPTER_TYPE_TO_STRING(MODEL_OUTPUT);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string OperationTypeToString(
    NNAdapterOperationType type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(ABS);
    NNADAPTER_TYPE_TO_STRING(ADAPTIVE_AVERAGE_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(ADAPTIVE_MAX_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(ADD);
    NNADAPTER_TYPE_TO_STRING(AND);
    NNADAPTER_TYPE_TO_STRING(ARG_MAX);
    NNADAPTER_TYPE_TO_STRING(ARG_MIN);
    NNADAPTER_TYPE_TO_STRING(ASSIGN);
    NNADAPTER_TYPE_TO_STRING(AVERAGE_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(BATCH_NORMALIZATION);
    NNADAPTER_TYPE_TO_STRING(CAST);
    NNADAPTER_TYPE_TO_STRING(CLIP);
    NNADAPTER_TYPE_TO_STRING(CHANNEL_SHUFFLE);
    NNADAPTER_TYPE_TO_STRING(CONCAT);
    NNADAPTER_TYPE_TO_STRING(CONV_2D);
    NNADAPTER_TYPE_TO_STRING(CONV_2D_TRANSPOSE);
    NNADAPTER_TYPE_TO_STRING(CUM_SUM);
    NNADAPTER_TYPE_TO_STRING(DEFORMABLE_CONV_2D);
    NNADAPTER_TYPE_TO_STRING(DEQUANTIZE);
    NNADAPTER_TYPE_TO_STRING(DIV);
    NNADAPTER_TYPE_TO_STRING(EQUAL);
    NNADAPTER_TYPE_TO_STRING(EXP);
    NNADAPTER_TYPE_TO_STRING(EXPAND);
    NNADAPTER_TYPE_TO_STRING(FILL);
    NNADAPTER_TYPE_TO_STRING(FILL_LIKE);
    NNADAPTER_TYPE_TO_STRING(FLATTEN);
    NNADAPTER_TYPE_TO_STRING(FLOOR);
    NNADAPTER_TYPE_TO_STRING(FULLY_CONNECTED);
    NNADAPTER_TYPE_TO_STRING(GATHER);
    NNADAPTER_TYPE_TO_STRING(GELU);
    NNADAPTER_TYPE_TO_STRING(GREATER);
    NNADAPTER_TYPE_TO_STRING(GREATER_EQUAL);
    NNADAPTER_TYPE_TO_STRING(GRID_SAMPLE);
    NNADAPTER_TYPE_TO_STRING(GROUP_NORMALIZATION);
    NNADAPTER_TYPE_TO_STRING(HARD_SIGMOID);
    NNADAPTER_TYPE_TO_STRING(HARD_SWISH);
    NNADAPTER_TYPE_TO_STRING(INSTANCE_NORMALIZATION);
    NNADAPTER_TYPE_TO_STRING(LAYER_NORMALIZATION);
    NNADAPTER_TYPE_TO_STRING(LEAKY_RELU);
    NNADAPTER_TYPE_TO_STRING(LESS);
    NNADAPTER_TYPE_TO_STRING(LESS_EQUAL);
    NNADAPTER_TYPE_TO_STRING(LOG);
    NNADAPTER_TYPE_TO_STRING(LOG_SOFTMAX);
    NNADAPTER_TYPE_TO_STRING(LP_NORMALIZATION);
    NNADAPTER_TYPE_TO_STRING(MAT_MUL);
    NNADAPTER_TYPE_TO_STRING(MAX);
    NNADAPTER_TYPE_TO_STRING(MAX_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(MESHGRID);
    NNADAPTER_TYPE_TO_STRING(MIN);
    NNADAPTER_TYPE_TO_STRING(MUL);
    NNADAPTER_TYPE_TO_STRING(NON_MAX_SUPPRESSION);
    NNADAPTER_TYPE_TO_STRING(NOT);
    NNADAPTER_TYPE_TO_STRING(NOT_EQUAL);
    NNADAPTER_TYPE_TO_STRING(PAD);
    NNADAPTER_TYPE_TO_STRING(PRIOR_BOX);
    NNADAPTER_TYPE_TO_STRING(POW);
    NNADAPTER_TYPE_TO_STRING(PRELU);
    NNADAPTER_TYPE_TO_STRING(QUANTIZE);
    NNADAPTER_TYPE_TO_STRING(RELU);
    NNADAPTER_TYPE_TO_STRING(RELU6);
    NNADAPTER_TYPE_TO_STRING(RANGE);
    NNADAPTER_TYPE_TO_STRING(REDUCE_MEAN);
    NNADAPTER_TYPE_TO_STRING(REDUCE_SUM);
    NNADAPTER_TYPE_TO_STRING(RESHAPE);
    NNADAPTER_TYPE_TO_STRING(RESIZE_NEAREST);
    NNADAPTER_TYPE_TO_STRING(RESIZE_LINEAR);
    NNADAPTER_TYPE_TO_STRING(ROI_ALIGN);
    NNADAPTER_TYPE_TO_STRING(SHAPE);
    NNADAPTER_TYPE_TO_STRING(SIGMOID);
    NNADAPTER_TYPE_TO_STRING(SLICE);
    NNADAPTER_TYPE_TO_STRING(STACK);
    NNADAPTER_TYPE_TO_STRING(SOFTMAX);
    NNADAPTER_TYPE_TO_STRING(SOFTPLUS);
    NNADAPTER_TYPE_TO_STRING(SPLIT);
    NNADAPTER_TYPE_TO_STRING(SQUARE);
    NNADAPTER_TYPE_TO_STRING(SQUEEZE);
    NNADAPTER_TYPE_TO_STRING(SUB);
    NNADAPTER_TYPE_TO_STRING(SUM);
    NNADAPTER_TYPE_TO_STRING(SWISH);
    NNADAPTER_TYPE_TO_STRING(TANH);
    NNADAPTER_TYPE_TO_STRING(TILE);
    NNADAPTER_TYPE_TO_STRING(TOP_K);
    NNADAPTER_TYPE_TO_STRING(TRANSPOSE);
    NNADAPTER_TYPE_TO_STRING(YOLO_BOX);
    NNADAPTER_TYPE_TO_STRING(UNSQUEEZE);
    NNADAPTER_TYPE_TO_STRING(UNSTACK);
    NNADAPTER_TYPE_TO_STRING(WHERE);
    default:
      name = type < 0 ? string_format("CUSTOM(type=%d)", type) : "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string FuseCodeToString(NNAdapterFuseCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(FUSED_NONE);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU1);
    NNADAPTER_TYPE_TO_STRING(FUSED_RELU6);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string DimensionsToString(const int32_t* dimensions_data,
                                                uint32_t dimensions_count) {
  std::string text;
  if (dimensions_count >= 1) {
    text = string_format("%d", dimensions_data[0]);
    for (uint32_t i = 1; i < dimensions_count; i++) {
      text += string_format(",%d", dimensions_data[i]);
    }
  }
  return text;
}

NNADAPTER_EXPORT std::string DeviceCodeToString(NNAdapterDeviceCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(CPU);
    NNADAPTER_TYPE_TO_STRING(GPU);
    NNADAPTER_TYPE_TO_STRING(ACCELERATOR);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

NNADAPTER_EXPORT std::string AutoPadCodeToString(NNAdapterAutoPadCode type) {
  std::string name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(AUTO_PAD_NONE);
    NNADAPTER_TYPE_TO_STRING(AUTO_PAD_SAME);
    NNADAPTER_TYPE_TO_STRING(AUTO_PAD_VALID);
    default:
      name = "UNKNOWN";
      break;
  }
  return name;
}

#undef NNADAPTER_TYPE_TO_STRING

NNADAPTER_EXPORT std::string OperandPrecisionCodeToSymbol(
    NNAdapterOperandPrecisionCode type) {
#define NNADAPTER_TYPE_TO_STRING(type, name) \
  case NNADAPTER_##type:                     \
    return #name;
  switch (type) {
    NNADAPTER_TYPE_TO_STRING(BOOL8, b);
    NNADAPTER_TYPE_TO_STRING(INT8, i8);
    NNADAPTER_TYPE_TO_STRING(UINT8, u8);
    NNADAPTER_TYPE_TO_STRING(INT16, i16);
    NNADAPTER_TYPE_TO_STRING(UINT16, u16);
    NNADAPTER_TYPE_TO_STRING(INT32, i32);
    NNADAPTER_TYPE_TO_STRING(UINT32, u32);
    NNADAPTER_TYPE_TO_STRING(INT64, i64);
    NNADAPTER_TYPE_TO_STRING(UINT64, u64);
    NNADAPTER_TYPE_TO_STRING(FLOAT16, f16);
    NNADAPTER_TYPE_TO_STRING(FLOAT32, f32);
    NNADAPTER_TYPE_TO_STRING(FLOAT64, f64);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT8_SYMM_PER_LAYER, qi8sl);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT8_SYMM_PER_CHANNEL, qi8sc);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT8_ASYMM_PER_LAYER, qu8al);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT16_SYMM_PER_LAYER, qi16sl);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT16_SYMM_PER_CHANNEL, qi16sc);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT16_ASYMM_PER_LAYER, qu16al);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT32_SYMM_PER_LAYER, qi32sl);
    NNADAPTER_TYPE_TO_STRING(QUANT_INT32_SYMM_PER_CHANNEL, qi32sc);
    NNADAPTER_TYPE_TO_STRING(QUANT_UINT32_ASYMM_PER_LAYER, qu32al);
    default:
      NNADAPTER_LOG(FATAL) << "Unhandle case: type="
                           << OperandPrecisionCodeToString(type) << ".";
      break;
  }
#undef NNADAPTER_TYPE_TO_STRING
  return 0;
}

NNADAPTER_EXPORT std::string OperandToString(core::Operand* operand) {
  return operand ? (OperandIdToString(operand) + "\n" +
                    OperandTypeToString(&operand->type))
                 : "nullptr";
}

NNADAPTER_EXPORT std::string OperandIdToString(core::Operand* operand) {
  return string_format("0x%X", reinterpret_cast<int64_t>(operand));
}

NNADAPTER_EXPORT std::string OperandValueToString(core::Operand* operand) {
  auto label = OperandIdToString(operand);
  auto& type = operand->type;
  auto buffer = operand->buffer;
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  auto is_scalar = type.dimensions.count == 0;
  auto is_vector = type.dimensions.count == 1;
  // Only peek the value from the constant scalar operand
  if (is_constant && is_scalar) {
#define OPERAND_SCALAR_VALUE_TO_STRING(ntype, dtype, dspecifier)               \
  case NNADAPTER_##ntype:                                                      \
    label = string_format("%" #dspecifier, *reinterpret_cast<dtype*>(buffer)); \
    break;
    switch (type.precision) {
      OPERAND_SCALAR_VALUE_TO_STRING(BOOL8, bool, d);
      OPERAND_SCALAR_VALUE_TO_STRING(INT8, int8_t, d);
      OPERAND_SCALAR_VALUE_TO_STRING(UINT8, uint8_t, u);
      OPERAND_SCALAR_VALUE_TO_STRING(INT16, int16_t, d);
      OPERAND_SCALAR_VALUE_TO_STRING(UINT16, uint16_t, u);
      OPERAND_SCALAR_VALUE_TO_STRING(INT32, int32_t, d);
      OPERAND_SCALAR_VALUE_TO_STRING(UINT32, uint32_t, u);
      OPERAND_SCALAR_VALUE_TO_STRING(INT64, int64_t, lld);
      OPERAND_SCALAR_VALUE_TO_STRING(UINT64, uint64_t, lld);
      OPERAND_SCALAR_VALUE_TO_STRING(FLOAT16, int16_t, d);
      OPERAND_SCALAR_VALUE_TO_STRING(FLOAT32, float, f);
      OPERAND_SCALAR_VALUE_TO_STRING(FLOAT64, double, f);
      default:
        NNADAPTER_LOG(FATAL) << "Can't peek the scalar value for "
                             << OperandPrecisionCodeToString(type.precision)
                             << ".";
        break;
    }
#undef OPERAND_SCALAR_VALUE_TO_STRING
  } else {
    if (is_constant && is_vector) {
      auto count = type.dimensions.data[0];
      if (count > 0 && count <= 4) {
#define OPERAND_VECTOR_VALUE_TO_STRING(ntype, dtype, dspecifier)               \
  case NNADAPTER_##ntype:                                                      \
    label +=                                                                   \
        string_format("%" #dspecifier, (reinterpret_cast<dtype*>(buffer))[0]); \
    for (uint32_t i = 1; i < count; i++) {                                     \
      label += string_format(",%" #dspecifier,                                 \
                             (reinterpret_cast<dtype*>(buffer))[i]);           \
    }                                                                          \
    break;
        label = "{";
        switch (type.precision) {
          OPERAND_VECTOR_VALUE_TO_STRING(BOOL8, bool, d);
          OPERAND_VECTOR_VALUE_TO_STRING(INT8, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(QUANT_INT8_SYMM_PER_LAYER, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_INT8_SYMM_PER_CHANNEL, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(UINT8, uint8_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_UINT8_ASYMM_PER_LAYER, uint8_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(INT16, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_INT16_SYMM_PER_LAYER, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_INT16_SYMM_PER_CHANNEL, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(UINT16, uint16_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_UINT16_ASYMM_PER_LAYER, uint16_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(INT32, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_INT32_SYMM_PER_LAYER, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_INT32_SYMM_PER_CHANNEL, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(UINT32, uint32_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(
              QUANT_UINT32_ASYMM_PER_LAYER, uint32_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(INT64, int64_t, lld);
          OPERAND_VECTOR_VALUE_TO_STRING(UINT64, uint64_t, lld);
          OPERAND_VECTOR_VALUE_TO_STRING(FLOAT16, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(FLOAT32, float, f);
          OPERAND_VECTOR_VALUE_TO_STRING(FLOAT64, double, f);
          default:
            NNADAPTER_LOG(FATAL) << "Can't peek the vector value for "
                                 << OperandPrecisionCodeToString(type.precision)
                                 << ".";
            break;
        }
        label += "}";
      }
#undef OPERAND_VECTOR_VALUE_TO_STRING
    }
    // Dimensions2String
    label += ":[" +
             DimensionsToString(type.dimensions.data, type.dimensions.count) +
             "]";
  }
  return string_format("%s:%s",
                       label.c_str(),
                       OperandPrecisionCodeToSymbol(type.precision).c_str());
}

NNADAPTER_EXPORT std::string OperandTypeToString(NNAdapterOperandType* type) {
  std::ostringstream os;
  os << " precision: " << OperandPrecisionCodeToString(type->precision)
     << std::endl;
  os << " layout: " << OperandLayoutCodeToString(type->layout) << std::endl;
  os << " lifetime: " << OperandLifetimeCodeToString(type->lifetime)
     << std::endl;
  os << " dimensions: [";
  for (uint32_t i = 0; i < type->dimensions.count; i++) {
    os << type->dimensions.data[i] << ",";
  }
  os << "]" << std::endl;
  switch (type->precision) {
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER: {
      os << " scale: " << type->symm_per_layer_params.scale;
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL: {
      os << " scales: [";
      for (uint32_t i = 0; i < type->symm_per_channel_params.scale_count; i++) {
        os << type->symm_per_channel_params.scales[i] << ",";
      }
      os << "]";
      os << " channel_dim: " << type->symm_per_channel_params.channel_dim;
    } break;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
    case NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER: {
      os << " scale: " << type->asymm_per_layer_params.scale;
      os << " zero_point: " << type->asymm_per_layer_params.zero_point;
    } break;
    default:
      break;
  }
  return os.str();
}

NNADAPTER_EXPORT std::string OperationIdToString(core::Operation* operation) {
  return string_format("@0x%X", reinterpret_cast<int64_t>(operation));
}

}  // namespace nnadapter
