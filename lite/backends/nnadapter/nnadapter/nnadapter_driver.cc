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

#include "nnadapter_driver.h"  // NOLINT
#include <stdarg.h>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
#include "nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {

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

std::string string_format(const std::string fmt_str, ...) {
  // Reserve two times as much as the length of the fmt_str
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    // Wrap the plain char array into the unique_ptr
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

std::string OperationTypeToString(NNAdapterOperationType type) {
#define OPERATION_TYPE_TO_STRING(type) \
  case NNADAPTER_##type:               \
    name = #type;                      \
    break;

  std::string name;
  switch (type) {
    OPERATION_TYPE_TO_STRING(CONV_2D);
    OPERATION_TYPE_TO_STRING(FULLY_CONNECTED);
    OPERATION_TYPE_TO_STRING(SIGMOID);
    OPERATION_TYPE_TO_STRING(RELU);
    OPERATION_TYPE_TO_STRING(RELU6);
    OPERATION_TYPE_TO_STRING(TANH);
    OPERATION_TYPE_TO_STRING(SOFTMAX);
    OPERATION_TYPE_TO_STRING(AVERAGE_POOL_2D);
    OPERATION_TYPE_TO_STRING(MAX_POOL_2D);
    OPERATION_TYPE_TO_STRING(ADD);
    OPERATION_TYPE_TO_STRING(SUB);
    OPERATION_TYPE_TO_STRING(MUL);
    OPERATION_TYPE_TO_STRING(DIV);
    default:
      name = "UNKNOWN";
      break;
  }

#undef OPERATION_TYPE_TO_STRING
  return name;
}

std::string Visualize(Model* model) {
#define APPEND_OPERAND_NODE()                                     \
  auto operand_name =                                             \
      string_format("@0x%X", reinterpret_cast<int64_t>(operand)); \
  if (!visited_operands.count(operand)) {                         \
    dot.AddNode(operand_name, {});                                \
    visited_operands.insert(operand);                             \
  }                                                               \
  std::vector<Dot::Attr> attrs;                                   \
  attrs.emplace_back("label", string_format("%d", i));

  Dot dot;
  std::ostringstream os;
  auto operations = SortOperationsInTopologicalOrder(model);
  std::set<Operand*> visited_operands;
  for (auto* operation : operations) {
    std::string operation_name = OperationTypeToString(operation->type);
    operation_name = string_format("%s@0x%X",
                                   operation_name.c_str(),
                                   reinterpret_cast<int64_t>(operation));
    dot.AddNode(operation_name,
                {Dot::Attr("shape", "box"),
                 Dot::Attr("style", "filled"),
                 Dot::Attr("color", "black"),
                 Dot::Attr("fillcolor", "yellow")});
    size_t input_count = operation->input_operands.size();
    for (size_t i = 0; i < input_count; i++) {
      auto* operand = operation->input_operands[i];
      APPEND_OPERAND_NODE()
      dot.AddEdge(operand_name, operation_name, attrs);
    }
    size_t output_count = operation->output_operands.size();
    for (size_t i = 0; i < output_count; i++) {
      auto* operand = operation->output_operands[i];
      APPEND_OPERAND_NODE()
      dot.AddEdge(operation_name, operand_name, attrs);
    }
  }
  os << dot.Build();

#undef APPEND_OPERAND_NODE
  return os.str();
}

std::vector<Operation*> SortOperationsInTopologicalOrder(Model* model) {
  std::vector<Operation*> operations;  // Operations in topological order
  std::vector<Operation*> queue;
  // Use to find all of adjacent operations according to a given operand.
  std::multimap<Operand*, Operation*> map;
  // The counters of variable inputs for all of operations.
  std::map<Operation*, uint32_t> counts;
  for (auto& operation : model->operations) {
    uint32_t count = 0;
    for (auto operand : operation.input_operands) {
      auto lifetime = operand->type.lifetime;
      if (lifetime == NNADAPTER_TEMPORARY_VARIABLE ||
          lifetime == NNADAPTER_MODEL_OUTPUT) {
        count++;
        map.insert(std::pair<Operand*, Operation*>(operand, &operation));
      }
    }
    if (count == 0) {
      // The operation which only depends the model inputs and constants
      queue.push_back(&operation);
    }
    counts[&operation] = count;
  }
  while (queue.size() > 0) {
    auto operation = queue.back();
    queue.pop_back();
    operations.push_back(operation);
    for (auto operand : operation->output_operands) {
      auto range = map.equal_range(operand);
      for (auto i = range.first; i != range.second; i++) {
        uint32_t& count = counts[i->second];
        if (--count == 0) {
          queue.push_back(i->second);
        }
      }
    }
  }
  return operations;
}

}  // namespace driver
}  // namespace nnadapter
