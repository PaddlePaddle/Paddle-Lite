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

#include <bmcompiler_defs.h>
#include <bmcompiler_if.h>
#include <bmcompiler_if_lite.h>
#include <bmcompiler_op_code.h>
#include <cmath>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

float* compute_elementwise_both_const(OpLite* op) {
  auto op_info = op->op_info();
  auto scope = op->scope();
  auto op_type = op_info->Type();

  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto y_var_name = op_info->Input("Y").front();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto y_dims = y->dims();
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  CHECK(output->data_size());
  CHECK_GT(SIZE_MAX / sizeof(float), output->data_size());
  float* cpu_data =
      static_cast<float*>(malloc(sizeof(float) * output->data_size()));
  CHECK(cpu_data != nullptr);
  CHECK_EQ(x_dims.size(), y_dims.size());
  const float* y_data = const_cast<const float*>(y->mutable_data<float>());
  const float* x_data = const_cast<const float*>(x->mutable_data<float>());
  if (op_type == "elementwise_mul") {
    for (size_t i = 0; i < output->data_size(); i++) {
      cpu_data[i] = x_data[i] * y_data[i];
    }
  } else if (op_type == "elementwise_add") {
    for (size_t i = 0; i < output->data_size(); i++) {
      cpu_data[i] = x_data[i] + y_data[i];
    }
  } else if (op_type == "elementwise_sub") {
    for (size_t i = 0; i < output->data_size(); i++) {
      cpu_data[i] = x_data[i] - y_data[i];
    }
  } else if (op_type == "elementwise_div") {
    for (size_t i = 0; i < output->data_size(); i++) {
      CHECK(fabsf(y_data[i]) > 1e-6f) << "y_data can not equal zero";
      cpu_data[i] = x_data[i] / y_data[i];
    }
  }
  return cpu_data;
}

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  // input
  const int input_num = 2;
  int** shape = new int*[input_num];
  int* dim = new int[input_num];
  const char** name = new const char*[input_num];
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  name[0] = static_cast<const char*>(x_var_name.c_str());
  dim[0] = x_dims.size();
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_dims[i]);
  }
  shape[0] = &i_x_shape_data[0];
  bool x_is_const = !graph->HasNode(x_var_name);
  auto y_var_name = op_info->Input("Y").front();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto y_dims = y->dims();
  name[1] = static_cast<const char*>(y_var_name.c_str());
  dim[1] = y_dims.size();
  std::vector<int32_t> i_y_shape_data(y_dims.size());
  for (size_t i = 0; i < y_dims.size(); i++) {
    i_y_shape_data[i] = static_cast<int>(y_dims[i]);
  }
  shape[1] = &i_y_shape_data[0];
  bool y_is_const = !graph->HasNode(y_var_name);
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  auto axis = op_info->GetAttr<int>("axis");
  int op_code{-1};
  if (op_type == "elementwise_mul") {
    op_code = BINARY_MUL;
  } else if (op_type == "elementwise_add") {
    op_code = BINARY_ADD;
  } else if (op_type == "elementwise_sub") {
    op_code = BINARY_SUB;
  } else if (op_type == "elementwise_div") {
    op_code = BINARY_DIV;
  } else {
    LOG(FATAL) << "UNSUPPORTED ELTWISE OPERATION: " << op_type;
  }
  const float* y_data = const_cast<const float*>(y->mutable_data<float>());
  const float* x_data = const_cast<const float*>(x->mutable_data<float>());
  auto unique_op_name = lite::subgraph::bm::UniqueName("expand_ndims");
  std::vector<int32_t> i_expand_shape_data;
  if (x_is_const && y_is_const) {
    float* cpu_data = compute_elementwise_both_const(op);
    bm_add_const_tensor(graph->GetCompilerHandle(),
                        static_cast<const char*>(output_var_name.c_str()),
                        const_cast<const int*>(&i_output_shape_data[0]),
                        output_dims.size(),
                        static_cast<bm_data_type_t>(DTYPE_FP32),
                        static_cast<const void*>(cpu_data));
  } else {
    if (y_is_const) {
      if (dim[0] == dim[1] || 2 == dim[0]) {
        bm_add_const_tensor(graph->GetCompilerHandle(),
                            name[1],
                            shape[1],
                            dim[1],
                            static_cast<bm_data_type_t>(DTYPE_FP32),
                            static_cast<const void*>(y_data));
      } else if (1 == dim[1] && 1 == axis) {
        add_expand_ndims_layer(
            graph->GetCompilerHandle(),
            name[1],
            shape[1],
            dim[1],
            static_cast<const float*>(y_data),
            -1,
            2,
            static_cast<const char*>(unique_op_name.c_str()));
        name[1] = static_cast<const char*>(unique_op_name.c_str());
        dim[1] = 3;
        i_expand_shape_data.push_back(i_y_shape_data[0]);
        i_expand_shape_data.push_back(1);
        i_expand_shape_data.push_back(1);
        shape[1] = &i_expand_shape_data[0];
        y_data = nullptr;
      }
    } else {
      if (dim[1] < dim[0]) {
        for (size_t i = 0; i < dim[1]; i++) {
          i_expand_shape_data.push_back(i_y_shape_data[i]);
        }
        for (size_t i = dim[1]; i < dim[0]; i++) {
          i_expand_shape_data.push_back(1);
        }
        add_reshape_layer_v2(graph->GetCompilerHandle(),
                             name[1],
                             shape[1],
                             dim[1],
                             static_cast<const char*>(unique_op_name.c_str()),
                             const_cast<const int*>(&i_expand_shape_data[0]),
                             i_expand_shape_data.size());
        dim[1] = dim[0];
        shape[1] = &i_expand_shape_data[0];
        name[1] = static_cast<const char*>(unique_op_name.c_str());
      }
    }
    add_binary_layer_v2(graph->GetCompilerHandle(),
                        name[0],
                        shape[0],
                        dim[0],
                        0,
                        static_cast<const float*>(x_data),
                        name[1],
                        shape[1],
                        dim[1],
                        0,
                        static_cast<const float*>(y_data),
                        static_cast<const char*>(output_var_name.c_str()),
                        op_code);
  }
  delete[] shape;
  delete[] name;
  delete[] dim;
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kBM,
                         paddle::lite::subgraph::bm::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_mul,
                         kBM,
                         paddle::lite::subgraph::bm::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_sub,
                         kBM,
                         paddle::lite::subgraph::bm::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_div,
                         kBM,
                         paddle::lite::subgraph::bm::ElementwiseConverter);
