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

NNADAPTER_EXPORT std::string Visualize(hal::Model* model) {
#define APPEND_OPERAND_NODE(mode)                                           \
  auto operand_id = OperandIdToString(operand);                             \
  auto operand_label = OperandValueToString(operand);                       \
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
  std::set<hal::Operand*> visited_operands;
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
      case NNADAPTER_CONCAT: {
        input_args.resize(input_count);
        for (int i = 0; i < input_count - 1; i++) {
          input_args[i] = string_format("input%d", i);
        }
        input_args[input_count - 1] = "axis";
        output_args = {"output"};
      } break;
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_TANH:
        input_args = {"input"};
        output_args = {"output"};
        break;
      case NNADAPTER_RESHAPE:
        input_args = {"input", "shape"};
        output_args = {"output"};
        break;
      case NNADAPTER_SOFTMAX:
        input_args = {"input", "axis"};
        output_args = {"output"};
        break;
      case NNADAPTER_CONV_2D:
        input_args = {"input",
                      "filter",
                      "bias",
                      "padding_left",
                      "padding_right",
                      "padding_top",
                      "padding_bottom",
                      "stride_width",
                      "stride_height",
                      "group",
                      "fuse_code",
                      "dilation_width",
                      "dilation_height"};
        output_args = {"output"};
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        input_args = {"input",
                      "padding_left",
                      "padding_right",
                      "padding_top",
                      "padding_bottom",
                      "stride_width",
                      "stride_height",
                      "filter_width",
                      "filter_height",
                      "fuse_code",
                      "ceil_mode",
                      "count_include_pad"};
        output_args = {"output"};
        break;
      case NNADAPTER_FULLY_CONNECTED:
        input_args = {"input", "weight", "bias", "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_SUB:
      case NNADAPTER_MUL:
      case NNADAPTER_DIV:
        input_args = {"input0", "input1", "fuse_code"};
        output_args = {"output"};
        break;
      case NNADAPTER_TRANSPOSE:
        input_args = {"input", "perm"};
        output_args = {"output"};
        break;
      default:
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
    NNADAPTER_TYPE_TO_STRING(TENSOR_BOOL8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT8);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_INT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_UINT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT16);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT32);
    NNADAPTER_TYPE_TO_STRING(TENSOR_FLOAT64);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT8_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT32_SYMM_PER_LAYER);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL);
    NNADAPTER_TYPE_TO_STRING(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER);
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
    NNADAPTER_TYPE_TO_STRING(ADD);
    NNADAPTER_TYPE_TO_STRING(AVERAGE_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(CONCAT);
    NNADAPTER_TYPE_TO_STRING(CONV_2D);
    NNADAPTER_TYPE_TO_STRING(DIV);
    NNADAPTER_TYPE_TO_STRING(FULLY_CONNECTED);
    NNADAPTER_TYPE_TO_STRING(HARD_SIGMOID);
    NNADAPTER_TYPE_TO_STRING(HARD_SWISH);
    NNADAPTER_TYPE_TO_STRING(MAX_POOL_2D);
    NNADAPTER_TYPE_TO_STRING(MUL);
    NNADAPTER_TYPE_TO_STRING(RELU);
    NNADAPTER_TYPE_TO_STRING(RELU6);
    NNADAPTER_TYPE_TO_STRING(RESHAPE);
    NNADAPTER_TYPE_TO_STRING(SIGMOID);
    NNADAPTER_TYPE_TO_STRING(SOFTMAX);
    NNADAPTER_TYPE_TO_STRING(SUB);
    NNADAPTER_TYPE_TO_STRING(TANH);
    NNADAPTER_TYPE_TO_STRING(TRANSPOSE);
    default:
      name = "UNKNOWN";
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

NNADAPTER_EXPORT std::string DimensionsToString(const int32_t* dimensions,
                                                uint32_t dimension_count) {
  std::string text;
  if (dimension_count >= 1) {
    text = string_format("%d", dimensions[0]);
    for (uint32_t i = 1; i < dimension_count; i++) {
      text += string_format(",%d", dimensions[i]);
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

#undef NNADAPTER_TYPE_TO_STRING

NNADAPTER_EXPORT int OperandPrecisionLength(
    NNAdapterOperandPrecisionCode type) {
#define NNADAPTER_PRECISION_LENGTH(type, bytes) \
  case NNADAPTER_##type:                        \
    return bytes;
  switch (type) {
    NNADAPTER_PRECISION_LENGTH(BOOL8, 1);
    NNADAPTER_PRECISION_LENGTH(INT8, 1);
    NNADAPTER_PRECISION_LENGTH(UINT8, 1);
    NNADAPTER_PRECISION_LENGTH(INT16, 2);
    NNADAPTER_PRECISION_LENGTH(UINT16, 2);
    NNADAPTER_PRECISION_LENGTH(INT32, 4);
    NNADAPTER_PRECISION_LENGTH(UINT32, 4);
    NNADAPTER_PRECISION_LENGTH(INT64, 8);
    NNADAPTER_PRECISION_LENGTH(UINT64, 8);
    NNADAPTER_PRECISION_LENGTH(FLOAT16, 2);
    NNADAPTER_PRECISION_LENGTH(FLOAT32, 4);
    NNADAPTER_PRECISION_LENGTH(FLOAT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_BOOL8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT8, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_INT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_UINT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT16, 2);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT32, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_FLOAT64, 8);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT8_SYMM_PER_LAYER, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER, 1);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT32_SYMM_PER_LAYER, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, 4);
    NNADAPTER_PRECISION_LENGTH(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER, 4);
    default:
      NNADAPTER_LOG(ERROR) << "Failed to get the length of "
                           << OperandPrecisionCodeToString(type) << ".";
      break;
  }
#undef NNADAPTER_PRECISION_LENGTH
  return 0;
}

NNADAPTER_EXPORT std::string OperandPrecisionName(
    NNAdapterOperandPrecisionCode type) {
#define NNADAPTER_PRECISION_NAME(type, name) \
  case NNADAPTER_##type:                     \
    return #name;
  switch (type) {
    NNADAPTER_PRECISION_NAME(BOOL8, b);
    NNADAPTER_PRECISION_NAME(INT8, i8);
    NNADAPTER_PRECISION_NAME(UINT8, u8);
    NNADAPTER_PRECISION_NAME(INT16, i16);
    NNADAPTER_PRECISION_NAME(UINT16, u16);
    NNADAPTER_PRECISION_NAME(INT32, i32);
    NNADAPTER_PRECISION_NAME(UINT32, u32);
    NNADAPTER_PRECISION_NAME(INT64, i64);
    NNADAPTER_PRECISION_NAME(UINT64, u64);
    NNADAPTER_PRECISION_NAME(FLOAT16, f16);
    NNADAPTER_PRECISION_NAME(FLOAT32, f32);
    NNADAPTER_PRECISION_NAME(FLOAT64, f64);
    NNADAPTER_PRECISION_NAME(TENSOR_BOOL8, b);
    NNADAPTER_PRECISION_NAME(TENSOR_INT8, i8);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT8, u8);
    NNADAPTER_PRECISION_NAME(TENSOR_INT16, i16);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT16, u16);
    NNADAPTER_PRECISION_NAME(TENSOR_INT32, i32);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT32, u32);
    NNADAPTER_PRECISION_NAME(TENSOR_INT64, i64);
    NNADAPTER_PRECISION_NAME(TENSOR_UINT64, u64);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT16, f16);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT32, f32);
    NNADAPTER_PRECISION_NAME(TENSOR_FLOAT64, f16);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT8_SYMM_PER_LAYER, qi8sl);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, qi8sc);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_UINT8_ASYMM_PER_LAYER, qu8al);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT32_SYMM_PER_LAYER, qi32sl);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, qi32sc);
    NNADAPTER_PRECISION_NAME(TENSOR_QUANT_UINT32_ASYMM_PER_LAYER, qu32al);
    default:
      NNADAPTER_LOG(ERROR) << "Failed to get the name of "
                           << OperandPrecisionCodeToString(type) << ".";
      break;
  }
#undef NNADAPTER_PRECISION_NAME
  return 0;
}

NNADAPTER_EXPORT std::string OperandToString(hal::Operand* operand) {
  return OperandIdToString(operand) + "\n" +
         OperandTypeToString(&operand->type);
}

NNADAPTER_EXPORT std::string OperandIdToString(hal::Operand* operand) {
  return string_format("@0x%X", reinterpret_cast<int64_t>(operand));
}

std::string OperandValueToString(hal::Operand* operand) {
  auto label = OperandIdToString(operand);
  auto& type = operand->type;
  auto buffer = operand->buffer;
  auto length = operand->length;
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  auto is_scalar = type.dimension_count == 0;
  auto is_vector = type.dimension_count == 1;
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
        NNADAPTER_LOG(ERROR) << "Can't peek the scalar value for "
                             << OperandPrecisionCodeToString(type.precision)
                             << ".";
        break;
    }
#undef OPERAND_SCALAR_VALUE_TO_STRING
  } else {
    if (is_constant && is_vector) {
      auto count = type.dimensions[0];
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
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_BOOL8, bool, d);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_INT8, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_INT8_SYMM_PER_LAYER, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_INT8_SYMM_PER_CHANNEL, int8_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_UINT8, uint8_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_UINT8_ASYMM_PER_LAYER, uint8_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_INT16, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_UINT16, uint16_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_INT32, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_INT32_SYMM_PER_LAYER, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, int32_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_UINT32, uint32_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(
              TENSOR_QUANT_UINT32_ASYMM_PER_LAYER, uint32_t, u);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_INT64, int64_t, lld);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_UINT64, uint64_t, lld);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_FLOAT16, int16_t, d);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_FLOAT32, float, f);
          OPERAND_VECTOR_VALUE_TO_STRING(TENSOR_FLOAT64, double, f);
          default:
            NNADAPTER_LOG(ERROR) << "Can't peek the vector value for "
                                 << OperandPrecisionCodeToString(type.precision)
                                 << ".";
            break;
        }
        label += "}";
      }
#undef OPERAND_VECTOR_VALUE_TO_STRING
    }
    // Dimensions2String
    label +=
        ":[" + DimensionsToString(type.dimensions, type.dimension_count) + "]";
  }
  return string_format(
      "%s:%s", label.c_str(), OperandPrecisionName(type.precision).c_str());
}

NNADAPTER_EXPORT std::string OperandTypeToString(NNAdapterOperandType* type) {
  const uint32_t max_scale_display_size = 20;
  std::ostringstream os;
  os << " precision: " << OperandPrecisionCodeToString(type->precision)
     << std::endl;
  os << " layout: " << OperandLayoutCodeToString(type->layout) << std::endl;
  os << " lifetime: " << OperandLifetimeCodeToString(type->lifetime)
     << std::endl;
  os << " dimensions: [";
  for (uint32_t i = 0; i < type->dimension_count; i++) {
    os << type->dimensions[i] << ",";
  }
  os << "]" << std::endl;
  switch (type->precision) {
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER: {
      os << " scale: " << type->symm_per_layer_params.scale;
    } break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL: {
      os << " scales: [";
      for (uint32_t i = 0; i < max_scale_display_size &&
                           i < type->symm_per_channel_params.scale_count;
           i++) {
        os << type->symm_per_channel_params.scales[i] << ",";
      }
      if (type->symm_per_channel_params.scale_count > max_scale_display_size) {
        os << "...";
      }
      os << "]";
      os << " channel_dim: " << type->symm_per_channel_params.channel_dim;
    } break;
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER: {
      os << " scale: " << type->asymm_per_layer_params.scale;
      os << " zero_point: " << type->asymm_per_layer_params.zero_point;
    } break;
    default:
      break;
  }
  return os.str();
}

NNADAPTER_EXPORT std::string OperationIdToString(hal::Operation* operation) {
  return string_format("@0x%X", reinterpret_cast<int64_t>(operation));
}

}  // namespace nnadapter
